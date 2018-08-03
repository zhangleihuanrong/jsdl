import { Backend } from '../backend';
import { ENV } from '../environments';
import { NDView as NdArray } from '../NdView/ndview';
import { Tensor } from '../tensor';
import { BackendTensor, DataType, Shape, TypedArray } from '../types';
import { assert as ASSERT } from '../utils/gadget';
import { WebGL2Driver } from './webgl/webgl2';
import { CoordinateMapping } from './webgl/coord2D';

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
  _texFlatLen: number; // it could be little smaller than W*H

  constructor(ndarr: NdArray, dtype: DataType = 'float32', texture: WebGLTexture = null, texShape: [number, number] = null, texFlatLen: number = 0) {
    this._dtype = dtype;
    this._array = ndarr;
    this._texture = texture;
    this._texShape = texShape;
    this._texFlatLen = texFlatLen;
  }

  MoveDataToGpu(webgl: WebGL2Driver): WebGLTensor {
    if (!this._texture) {
      ASSERT(this._array.data != null, "No CPU Data exists.")
      this._texShape = webgl.calc2DTextureSizeForFlatLen(this._array.data.length);
      this._texture = webgl.create2DGLTexture(this._array.data as TypedArray, this._texShape, this._dtype);
      this._texFlatLen = this._array.data.length;
      this._array.data = null;  // clear cpu data
    }
    return this;
  }

  PrepareGpuData(webgl: WebGL2Driver): WebGLTensor {
    if (!this._texture) {
      ASSERT(this._array.data == null, "cpu data already exists");
      // create texture for render into it, no cpu data should exists
      // infact, should only have shape and stride, others should be null or zero.
      this._texShape = webgl.calc2DTextureSizeForFlatLen(this.size());
      this._texture = webgl.create2DGLTexture(null, this._texShape, this._dtype);
      this._texFlatLen = this.size();
    }
    return this;
  }

  DumpDataFromGPU(webgl: WebGL2Driver): WebGLTensor {
    if (!this._array.data) {
      ASSERT(this._texShape != null, "No GPU data exists!");
      const ta = webgl.DumpTexture(this._texture, this._texShape, this._texFlatLen, this._dtype);
      this._array.data = ta;
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

    //hack to see texture works or not at all
    if (bt._array.data != null && bt._texture == null) {
      bt.MoveDataToGpu(this.webgl);
    }

    bt.DumpDataFromGPU(this.webgl);
    let ndx = bt._array;
    if (ndx) {
      return ndx.rebuild().data as TypedArray;
    }
    return null;
  }

  transpose(x: Tensor, perm: number[]): Tensor {
    const bt = backendTensorOf(x);
    const ndarr = bt._array.transpose(perm);
    return Tensor.fromBackend(new WebGLTensor(ndarr, bt._dtype, bt._texture, bt._texShape, bt._texFlatLen));
  }

  matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor {
    const A = backendTensorOf(a).MoveDataToGpu(this.webgl);
    const B = backendTensorOf(b).MoveDataToGpu(this.webgl);

    const shapeC = [a.shape[0], b.shape[1]];
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
  ivec2 A_size = textureSize(A, 0);
  ivec2 B_size = textureSize(B, 0);

  // should be output C's texW and C's texH
  int out_x = int(2.0 * outTex.x);
  int out_y = int(2.0 * outTex.y);
  int commonDim = ${A._array.shape[1]};

  float sum = 0.;
  for (int i = 0; i < commonDim; ++i) {
    float a = getA(out_y, i);
    float b = getB(i, out_x);
    sum += a * b;
  }

  outColor = vec4(sum);
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
    const ndarr = bt._array.reshape(newShape);
    return Tensor.fromBackend(new WebGLTensor(ndarr, bt._dtype, bt._texture, bt._texShape));
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
