import { Backend } from '../backend';
import { ENV } from '../environments';
import { NDView as NdArray } from '../NdView/ndview';
import { Tensor } from '../tensor';
import { BackendTensor, DataType, Shape, TypedArray } from '../types';
import { assert as ASSERT } from '../utils/gadget';
import { WebGL2Driver } from './webgl/webgl2';
import { CoordinateMapping } from './webgl/coord2D';
import { canBroadcastTo, getUnsqueezeAxisForBroadcast, getUnsqueezedShapeForBroadcast, getBroadcastRepeats } from '../utils/shapeTools';

export class WebGLTensor implements BackendTensor {
  // basic info, TODO: combine with _array: ndarray
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

  PrepareGpuData(webgl: WebGL2Driver): WebGLTensor {
    if (!this._texture) {
      ASSERT(this._array.data == null, "cpu data already exists");
      ASSERT(this._array.isOriginalCore(), "ndarray information should only mapping to original data!");
      // create texture for render into it, no cpu data should exists
      // infact, should only have shape and stride, others should be null or zero.
      this._texShape = webgl.calc2DTextureSizeForShape(this._array.coreShape);
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

  DumpDataFromGPU(webgl: WebGL2Driver): WebGLTensor {
    if (!this._array.data) {
      ASSERT(this._texture != null, "GPU data not exists!");
      this._array.data = webgl.DumpTexture(this._texture, this._texShape, this._array.dataLen, this._dtype);
    }
    return this;
  }

  shape(): number[] {
    return this._array.shape;
  }

  dtype(): DataType {
    return this._dtype;
  }

  size(): number {
    return this.shape().reduce((m, v) => m * v, 1);
  }
};


function backendTensorOf(t: Tensor): WebGLTensor {
  return (t.data) as WebGLTensor;
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
    t.data = backendTensor;
  }

  make(t: Tensor, dtype: DataType, shape: number[], values: TypedArray): void {
    t.data = new WebGLTensor(new NdArray(values, shape), dtype);
  }

  free(t: Tensor): void {
    delete t.data;
  }

  readSync(x: Tensor): TypedArray {
    const bt = backendTensorOf(x);

    // //hack to see texture works or not at all
    // if (bt._array.data != null && bt._texture == null) {
    //   bt.MoveDataToGpu(this.webgl);
    // }

    bt.DumpDataFromGPU(this.webgl);
    let ndx = bt._array;
    if (ndx) {
      return ndx.rebuild().data as TypedArray;
    }
    return null;
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

    if (transposeA == true && A.shape().length > 1) A = A.transpose();
    if (transposeB == true && B.shape().length > 1) B = B.transpose();

    if (A.shape().length == 1) A = A.expandDim(0);
    if (B.shape().length == 1) B = B.expandDim(1);

    const shapeAMul = A.shape().slice(A.shape().length - 2);
    const shapeBMul = B.shape().slice(B.shape().length - 2);
    ASSERT(shapeAMul[1] == shapeBMul[0], `shape[${shapeAMul}] can not matMul with shape[${shapeBMul}]`);
    const commonDim = shapeAMul[1];

    const shapeAPrefix = A.shape().slice(0, A.shape.length - 2);
    const shapeBPrefix = B.shape().slice(0, B.shape.length - 2);
    if (shapeAPrefix.length > 0) {
      ASSERT(canBroadcastTo(shapeAPrefix, shapeBPrefix), "Can not broadcast b to a");
      const unsq = getUnsqueezeAxisForBroadcast(shapeAPrefix, shapeBPrefix);
      if (unsq) B = B.unsqueeze(unsq);
      const shapeBExpanded = getUnsqueezedShapeForBroadcast(shapeAPrefix, shapeBPrefix);
      const repeats = getBroadcastRepeats(shapeAPrefix, shapeBExpanded);
      if (repeats) B = B.tile(repeats.concat(1, 1));
    }

    const shapeC = shapeAPrefix.concat(shapeAMul[0], shapeBMul[1]);
    const rankC = shapeC.length;
    const C = new WebGLTensor(new NdArray(null, shapeC)).PrepareGpuData(this.webgl);

    const code = `#version 300 es
precision highp float;
precision highp int;

in vec2 outTex;
uniform sampler2D A;
uniform sampler2D B;
out vec4 outColor;

${CoordinateMapping.glslGet(A, 'A')}

${CoordinateMapping.glslGet(B, 'B')}
 
void main() {
  ${CoordinateMapping.snippetLogicFormST(C, 'C', 'idx_', 'outTex')}
  
  float sum = 0.0;
  for (int k = 0; k < ${commonDim}; ++k) {  // length of the common axis
    float a = getA(${CoordinateMapping.argList(rankC, 'idx_', [rankC-1, 'k'])});
    float b = getB(${CoordinateMapping.argList(rankC, 'idx_', [rankC-2, 'k'])});
    sum += (a * b);
  }

  outColor = vec4(sum, 0.0, 0.0, 0.0);
}
`;
    console.log(code);
    const program = this.webgl.compileProgram(code);

    this.webgl.runProgram(
      program,
      C._texture,
      C._texShape,
      [{ name: 'A', texture: A._texture }, { name: 'B', texture: B._texture }], 
      null);

    return Tensor.fromBackend(C);
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
    // do it cpu currently, could do it in GPU later. 
    bt.DumpDataFromGPU(this.webgl);
    ndarr = bt._array.reshape(newShape);
    return Tensor.fromBackend(new WebGLTensor(ndarr, bt._dtype));
  }

  add(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not implemented');
  }

  neg(x: Tensor): Tensor {
    throw new Error('Not implemented');
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    throw new Error('Not implemented');
  }

  relu(x: Tensor): Tensor {
    throw new Error('Not implemented');
  }

  conv2d(
    x: Tensor, filter: Tensor, strides: number | [number, number],
    padding: number[], dataFormat: 'NHWC' | 'NCHW',
    dialations: number | [number, number]): Tensor {

      throw new Error('Method not implemented.');
  }
};


const backendName: string = 'Backend_WebGL';
const backendScore: number = 16;
const backendWebGl = new WebGLBackend();
console.info(JSON.stringify(backendWebGl));
if (backendWebGl.isSupported) {
  ENV.registerBackend(backendName, backendWebGl, backendScore);
}
