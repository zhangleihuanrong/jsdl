import { Backend } from '../backend';
import { ENV } from '../environments';
import { NDView as NdArray } from '../NdView/ndview';
import { Tensor } from '../tensor';
import { BackendTensor, DataType, Shape, TypedArray } from '../types';
import { assert as ASSERT } from '../utils/gadget';
import { WebGL2Driver } from './webgl/webgl2';
import { WebGlProgramMatMul } from './webgl/matMul';
import { WebGlProgramConv2d } from './webgl/conv2D';
import { WebGlProgramPad } from './webgl/padding';
import { WebGlProgramUnaryOp } from './webgl/unaryops';
import { WebGlProgramSum2D } from './webgl/sum';
import { WebGlProgramLogSumExp2D } from './webgl/logSumExp';

export class WebGLTensor implements BackendTensor {
  _dtype: DataType;

  _array: NdArray;

  // Other fields will be used by webgl
  // Note the flattened _texture data is just same as the _array.data,
  // which means we could do logic shape/transform/gather/padding/repeat/slice/step...
  // on the _array part with even when its data set to null.
  _texture: WebGLTexture;  // 2D Texture
  _texShape: [number, number]; // W*H

  constructor(ndarr: NdArray, dtype: DataType = 'float32', texture: WebGLTexture = null, texShape: [number, number] = null) {
    this._dtype = dtype;
    this._array = ndarr;
    this._texture = texture;
    this._texShape = texShape;
  }

  MoveDataToGpu(webgl: WebGL2Driver): WebGLTensor {
    if (!this._texture) {
      ASSERT(this._array.data != null, "No CPU Data exists.")
      this._texShape = (this._array.isOriginalCore()) ? webgl.calc2DTextureSizeForShape(this._array.coreShape) : webgl.calc2DTextureSizeForFlatLen(this._array.data.length);
      this._texture = webgl.create2DGLTexture(this._array.data as TypedArray, this._texShape, this._dtype);
      this._array.data = null;  // clear cpu data
    }
    return this;
  }

  calc2DTextureSize(webgl: WebGL2Driver) {
    this._texShape = webgl.calc2DTextureSizeForShape(this._array.coreShape);
  }

  PrepareGpuData(webgl: WebGL2Driver): WebGLTensor {
    if (!this._texture) {
      ASSERT(this._array.data == null, "cpu data already exists");
      ASSERT(this._array.isOriginalCore(), "ndarray information should only mapping to original data!");
      // create texture for render into it, no cpu data should exists
      // infact, should only have shape and stride, others should be null or zero.
      this.calc2DTextureSize(webgl);
      this._texture = webgl.create2DGLTexture(null, this._texShape, this._dtype);
    }
    return this;
  }

  transpose(perm?: number[]): WebGLTensor {
    const ndarr = this._array.transpose(perm);
    return new WebGLTensor(ndarr, this._dtype, this._texture, this._texShape);
  }

  expandDim(axis?: number) : WebGLTensor {
    const ndarr = this._array.expandDim(axis);
    return new WebGLTensor(ndarr, this._dtype, this._texture, this._texShape);
  }

  unsqueeze(axises : number[]) : WebGLTensor {
    const ndarr = this._array.unsqueeze(axises);
    return new WebGLTensor(ndarr, this._dtype, this._texture, this._texShape);
  }

  tile(reps: number[]) : WebGLTensor {
    const ndarr = this._array.tile(reps);
    return new WebGLTensor(ndarr, this._dtype, this._texture, this._texShape);
  }

  pick(indices: number[]) : WebGLTensor {
    const ndarr = this._array.pick(indices);
    return new WebGLTensor(ndarr, this._dtype, this._texture, this._texShape);
  }

  // This read array in original layout.
  DumpDataFromGPU(webgl: WebGL2Driver): TypedArray {
      ASSERT(this._texture != null, "GPU data not exists!");
      return webgl.DumpTexture(this._texture, this._texShape, this._array.dataLen, this._dtype);
  }

  // This read array in new layout, i.e., only coreShape/coreStride. Other fields are 0 or null or default.
  ReadFromGPU(webgl: WebGL2Driver): TypedArray {
    // TODO:
    return null;
  }

  get shape(): number[] {
    return this._array.shape;
  }

  get dtype(): DataType {
    return this._dtype;
  }

  get size(): number {
    return this.shape.reduce((m, v) => m * v, 1);
  }
};


function backendTensorOf(t: Tensor): WebGLTensor {
  return (t)? (t.data as WebGLTensor) : null;
}


class WebGLBackend implements Backend {
  isSupported: boolean = false;
  webgl: WebGL2Driver = null;

  constructor() {
    this.webgl = new WebGL2Driver();
    this.isSupported = this.webgl._isSupported;
  }

  wrap(t: Tensor, backendTensor: BackendTensor): void {
    if (!(backendTensor instanceof WebGLTensor)) {
      throw new Error('Not dealing with my own backend tensor!')
    }
    if (backendTensor._array.data != null) {
      backendTensor.MoveDataToGpu(this.webgl);
    }
    t.data = backendTensor;
  }

  make(t: Tensor, dtype: DataType, shape: number[], values: TypedArray): void {
    const bt = new WebGLTensor(new NdArray(values, shape), dtype);
    bt.MoveDataToGpu(this.webgl);
    t.data = bt;
  }

  free(t: Tensor): void {
    delete t.data;
  }

  read(x: Tensor): TypedArray {
    const bt = backendTensorOf(x);
    const ba = bt._array;

    const ta = bt.DumpDataFromGPU(this.webgl);
    let ndx = new NdArray(ta, ba.coreShape, ba.dataLen, ba.coreStride, ba.coreOffset, ba.gather, ba.repeat, ba.padding, ba.paddingValue);
    return ndx.rebuild().data as TypedArray;
  }

  transpose(x: Tensor, perm?: number[]): Tensor {
    return Tensor.fromBackend(backendTensorOf(x).transpose(perm));
  }

  tile(x: Tensor, repeats: number[]) : Tensor {
    return Tensor.fromBackend(backendTensorOf(x).tile(repeats));
  }

  pick(x: Tensor, indices: number[]) : Tensor {
    return Tensor.fromBackend(backendTensorOf(x).pick(indices));;
  }

  //
  // output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.
  matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor {
    ASSERT(a.shape.length >= 1 && b.shape.length >= 1, "Scala is not allowed in matMul");
    ASSERT(a.shape.length >= b.shape.length, "Too many dimensions on b");
    let A = backendTensorOf(a).MoveDataToGpu(this.webgl);
    let B = backendTensorOf(b).MoveDataToGpu(this.webgl);

    const prg = new WebGlProgramMatMul(this.webgl, {A, B, transposeA, transposeB});
    return Tensor.fromBackend(prg.run());
  }

  reshape(x: Tensor, newShape: Shape): Tensor {
    const bt = backendTensorOf(x);
    let ndarr = bt._array.reshape(newShape);
    if (ndarr) {
      if (ndarr.data === bt._array.data) {
        return Tensor.fromBackend(new WebGLTensor(ndarr, bt._dtype, bt._texture, bt._texShape));
      }
      return Tensor.fromBackend(new WebGLTensor(ndarr, bt._dtype));
    }
    // // do it cpu currently, could do it in GPU later. 
    // bt.DumpDataFromGPU(this.webgl);
    // ndarr = bt._array.reshape(newShape);
    // return Tensor.fromBackend(new WebGLTensor(ndarr, bt._dtype));
    return null;
  }

  add(a: Tensor, b: Tensor): Tensor {
    return Tensor.fromBackend(this.add_bk(backendTensorOf(a), backendTensorOf(b)));
  }

  add_bk(a: WebGLTensor, b: WebGLTensor): WebGLTensor {
    throw new Error('Not implemented');
  }

  neg(x: Tensor): Tensor {
    throw new Error('Not implemented');
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not implemented');
  }

  relu(x: Tensor): Tensor {
      return Tensor.fromBackend(this.relu_bk(backendTensorOf(x)));
  }

  relu_bk(btx: WebGLTensor) : WebGLTensor {
      const prg = new WebGlProgramUnaryOp(this.webgl, 'relu', btx);
      return prg.run();
  }

  pad(x: Tensor, paddings: [number, number][]) : Tensor {
    const btp = this.pad_bk(backendTensorOf(x), paddings);
    return Tensor.fromBackend(btp);
  }

  pad_bk(x: WebGLTensor, paddings: [number, number][]) : WebGLTensor {
    let padded = x._array.pad(paddings);
    if (padded == null) {
      const prg = new WebGlProgramPad(this.webgl, x, paddings);
      return prg.run();
    }
    return new WebGLTensor(padded, x._dtype, x._texture, x._texShape);
  }

  conv2d(
    x: Tensor, filter: Tensor, strides: number | [number, number],
    padding: number[], dataFormat: 'NHWC' | 'NCHW',  dilations: number | [number, number],
    groups: number = 1, bias: Tensor = null): Tensor {
      const br = this.conv2d_bk(backendTensorOf(x), backendTensorOf(filter), strides, padding, 
          dataFormat, dilations, groups, bias);
      return Tensor.fromBackend(br);
  }

  // btx: [N, C, H, W] or [N, H, W, C]
  // btk: [H, W, in, out] or [out, in, H, W]
  conv2d_bk(
    btx: WebGLTensor, btk: WebGLTensor, strides: number | [number, number],
    padding: number[], dataFormat: 'NHWC' | 'NCHW',  dilations: number | [number, number],
    groups: number = 1, bias: Tensor = null): WebGLTensor {

    ASSERT(btx.shape.length == 4 && btk.shape.length == 4, "Shape error input image or kernel");
    if (!(strides instanceof Array)) strides = [strides as number, strides as number];
    if (!(dilations instanceof Array)) dilations = [dilations as number, dilations as number];
    if (padding == null) padding = [0, 0, 0, 0];
    ASSERT(padding.length == 4 && padding.every(v => v >= 0 && (v == (v|0))), "padding values is wrong!");

    if (dataFormat == 'NCHW') {
      btx = btx.transpose([0, 2, 3, 1]);
      btk = btk.transpose([2, 3, 1, 0]);
    }

    const prg = new WebGlProgramConv2d(
      this.webgl, btx, btk, padding,
      strides, dilations, groups, 
      backendTensorOf(bias));

    let bt = prg.run();

    if (dataFormat == 'NCHW') {
      bt = bt.transpose([0, 3, 1, 2]);
    }
    return bt;
  }

  batchNormalize(x: Tensor, scale: Tensor, bias: Tensor, mean: Tensor, variance: Tensor, epsilon: number, momentum: number, spatial: number): Tensor {
    throw new Error("Method not implemented.");
  }
  maxPool(x: Tensor, kernelShape: number[], strides: number[], pads: number[], storageOrder: number): Tensor {
    throw new Error("Method not implemented.");
  }
  averagePool(x: Tensor, kernelShape: number[], strides: number[], pads: number[], count_in_pad: number): Tensor {
    throw new Error("Method not implemented.");
  }
  gemm(a: Tensor, b: Tensor, bias?: Tensor, alpha?: number, beta?: number, transposeA?: boolean, transposeB?: boolean): Tensor {
    throw new Error("Method not implemented.");
  }
  
  softmax(logits: Tensor, axis?: number): Tensor {
    throw new Error("Method not implemented.");
  }

  softmax_bk(logits: WebGLTensor, axis?: number) : WebGLTensor {
    const logSumExp = new WebGlProgramLogSumExp2D(this.webgl, logits);
    const lse = logSumExp.run();
    return null;
  }

  exp(x: Tensor): Tensor {
    return Tensor.fromBackend(this.exp_bk(backendTensorOf(x)));
  }

  exp_bk(btx: WebGLTensor) : WebGLTensor {
    const prg = new WebGlProgramUnaryOp(this.webgl, 'exp', btx);
    return prg.run();
  }

  sum(x: Tensor, axises?: number[], keepDims?: boolean): Tensor {
    return Tensor.fromBackend(this.sum_bt(backendTensorOf(x), axises, keepDims));
  }

  sum_bt(x: WebGLTensor, axises?: number[], keepDims?: boolean) : WebGLTensor {
    // suppose x is tensor 2d and reduce on dimension 1
    const prg = new WebGlProgramSum2D(this.webgl, x);
    return prg.run();
  }
};


const backendName: string = 'Backend_WebGL';
const backendScore: number = 16;
const backendWebGl = new WebGLBackend();
console.info(JSON.stringify(backendWebGl));
if (backendWebGl.isSupported) {
  ENV.registerBackend(backendName, backendWebGl, backendScore);
}
