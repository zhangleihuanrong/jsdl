import * as ndarray from 'ndarray';
import * as nd_gemm from 'ndarray-gemm';
import * as nd_ops from 'ndarray-ops';

import {Backend} from '../backend';
import {ENV} from '../environments';
import {Tensor} from '../tensor';
import {BackendTensor, DataType, Shape, TypedArray} from '../types';

const vertexShaderSource = `
#version 300 es
precision highp float;

in vec3 position;
in vec2 texcoord;
out vec2 outTex;

void main () {
  gl_Position = vec4(position, 1.0);
  outTex = texcoord;
}
`;


class WebGLTensor implements BackendTensor {
  //basic info, TODO: combine with _array: ndarray
  _dtype: DataType;

  // could be null
  _array: TypedArray;
  // final shape/strides for this tensor
  _shape: number[];
  _strides: number[];

  // Other fields will be used by webgl
  _origShape: number[];
  _texture: WebGLTexture;  // 2D Texture
  _texW: number;
  _texH: number;
  _axisX: number[];
  _axisY: number[];

  constructor(nda: ndarray, shape?: number[], dtype?: DataType) {
    if (nda) {
      this._array = nda;
      this._shape = nda.shape;
      this._dtype = nda.dtype as DataType;
    }
    else {
      // without data currently
      this._array = null;
      this._shape = shape;
      this._dtype = dtype;
    }
  }

  calulateTexture2DShape(MAX_TEXTURE_SIZE: number) : boolean {
    if (this._shape.length == 1) {
      this._texW = this._shape[0];
      this._axisX = [ 0 ];
      this._texH = 1;
      this._axisY = [ ];
    }
    else if (this._shape.length == 2) {
      this._texW = this._shape[1];
      this._axisX = [ 1 ];
      this._texH = this._shape[0];
      this._axisY = [ 0 ];
    }
    else {
      // TODO: current is hacky
      let m = 0;
      while (m < this._shape.length && this._shape[m] == 1) --m;

      this._texW = 1;
      this._axisX = [];
      let i = this._shape.length - 1;
      while (i > 0 && (i >= m || this._texW == 1) && this._texW * this._shape[i] <= MAX_TEXTURE_SIZE) {
        this._texW *= this._shape[i];
        this._axisX.unshift(i);
        --i;
      }
      this._texH = 1;
      this._axisY = [];
      for (; i >= 0; --i) {
        this._texH *= this._shape[i];
        this._axisY.unshift(i);
      }
    }
    return (this._texW <= MAX_TEXTURE_SIZE && 
            this._texH <= MAX_TEXTURE_SIZE &&
            this._axisX.length <= 4 && this._axisY.length <= 4);
  }

  shape(): number[] {
    return this._shape;
  }
  dtype(): DataType {
    return this._dtype;
  }
  size(): number {
    return this._shape.reduce((m, v) => m * v, 1);
  }
};

// TODO: move all ndarray related basic to seperate files.
function rangedArray(s: number): number[] {
  const arr = new Array(s);
  for (let i = 0; i < s; ++i) arr[i] = i;
  return arr;
}

function broadCastedNdarray(a: ndarray, newShape: number[]): ndarray {
  a = a.transpose(...rangedArray(a.shape.length));
  const shape: number[] = a.shape;
  const strides: number[] = a.stride;
  shape.forEach((orig, i) => {
    if (orig == 1 && newShape[i] > 1) {
      strides[i] = 0;
    }
  });
  a.shape = newShape;
  return a;
}


function ndarrayOf(t: Tensor): ndarray {
  return (t.data) ? (t.data as WebGLTensor)._array : null;
}

function backendTensorOf(t: Tensor) : WebGLTensor {
  return (t.data) as WebGLTensor;
}

class WebGLBackend implements Backend {
  _isSupported: boolean = false;
  _canvas: HTMLCanvasElement = null;
  _glContext: WebGL2RenderingContext = null;
  _vertexShader: WebGLShader = null;
  MAX_TEXTURE_SIZE: number = 0;
  MAX_TEXTURE_IMAGE_UNITS: number = 0;
  _refs = {textures: [], buffers: []};

  constructor() {
    if (typeof window !== 'undefined') {
      this._canvas = document.createElement('canvas');
      this._glContext = this._canvas.getContext('webgl2') as WebGL2RenderingContext;
      if (this._glContext) {
        const gl = this._glContext;
        this._isSupported = true;
        gl.getExtension('EXT_color_buffer_float');
        this.MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
        this.MAX_TEXTURE_IMAGE_UNITS = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
        this.createCommonVertexShader();
      } else {
        console.warn(
            'Unable to initialize WebGL2 -- your browser may not support it.');
      }
    }
  }

  /**
   * Creates and compiles passthrough vertex shader that we will attach
   * to all our programs.
   */
  createCommonVertexShader() {
    const gl = this._glContext

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);

    const success = gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS);
    if (!success) {
      console.error(gl.getShaderInfoLog(vertexShader));
      gl.deleteShader(vertexShader);
      this._isSupported = false;
    }

    this._vertexShader = vertexShader;
  }

  /**
   * Compiles fragment shader from source and creates program from it,
   * using our passthrough vertex shader.
   *
   * @param source - fragment shader GLSL source code
   * @returns WebGL program to be run later
   */
  compileProgram(source: string): WebGLProgram {
    const gl = this._glContext;

    // create and compile fragment shader
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, source);
    gl.compileShader(fragmentShader);

    let success = gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS);
    if (!success) {
      console.error(gl.getShaderInfoLog(fragmentShader));
      gl.deleteShader(fragmentShader);
      this._isSupported = false;
    }

    // create program and attach compiled shaders
    const program = gl.createProgram();
    gl.attachShader(program, this._vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    success = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (!success) {
      const msg = gl.getProgramInfoLog(program);
      console.error(msg);
      throw new Error(msg);
    }

    this.setupVertices(program);
    return program;
  }

  /**
   * Store reference to WebGL texture or buffer on class instance, useful for when we want to delete later
   */
  storeRef(texOrBuf : WebGLTexture|WebGLBuffer) {
    if (texOrBuf instanceof WebGLTexture) {
      this._refs.textures.push(texOrBuf);
    } else if (texOrBuf instanceof WebGLBuffer) {
      this._refs.buffers.push(texOrBuf);
    }
  }

  /**
   * Deletes all stored references to WebGL textures and buffers
   */
  clearRefs() {
    const gl = this._glContext
    this._refs.textures.forEach(texture => gl.deleteTexture(texture));
    this._refs.buffers.forEach(buffer => gl.deleteBuffer(buffer));
    this._refs = { textures: [], buffers: [] }
  }

  /**
   * Setup vertices
   *
   * @param {WebGLProgram} program
   */
  setupVertices(program) {
    const gl = this._glContext;

    const position = gl.getAttribLocation(program, 'position');
    const positionVertexObj = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionVertexObj);
    this.storeRef(positionVertexObj);

    const vs = new Float32Array([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]);
    gl.bufferData(gl.ARRAY_BUFFER, vs, gl.STATIC_DRAW);
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(position);

    const texcoord = gl.getAttribLocation(program, 'texcoord');
    const texcoordVertexObj = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texcoordVertexObj);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(texcoord, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(texcoord);
    this.storeRef(texcoordVertexObj);

    const indicesVertexObj = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indicesVertexObj);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    this.storeRef(indicesVertexObj);
  }

  wrap(t: Tensor, backendTensor: BackendTensor): void {
    if (!(backendTensor instanceof WebGLTensor)) {
      throw new Error('Not dealing with my own backend tensor!')
    }
    t.data = backendTensor;
  }

  make(t: Tensor, dtype: DataType, shape: number[], values: TypedArray): void {
    if (values) {
      t.data = new WebGLTensor(ndarray(values, shape));
    }
    else {
      t.data = new WebGLTensor(null, shape, dtype);
    }
  }

  free(t: Tensor): void {
    // TODO
    delete t.data;
  }


  /**
   * Bind output texture
   */
  bindOutputTexture(outputTexture: WebGLTexture, shape: number[], framebuffer?: WebGLFramebuffer) {
    const gl = this._glContext;
    gl.viewport(0, 0, shape[1], shape[0]);
    const activeFramebuffer = framebuffer || gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, activeFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);
    // check if you can read from this type of texture.
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('Can not read from this type of texture.')
    }
    if (!framebuffer) gl.deleteFramebuffer(activeFramebuffer);
  }

  create2DGLTexture(bt: WebGLTensor): WebGLTexture {
    const gl = this._glContext;

    if (!bt._texture && bt.calulateTexture2DShape(this.MAX_TEXTURE_SIZE)) {
      const texture = gl.createTexture();
      const tt = (bt._dtype == 'int32') ? gl.INT : ((bt._dtype == 'bool')? gl.BYTE : gl.FLOAT);
  
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, bt._texW, bt._texH, 0, gl.RED, tt, bt._array, 0);
  
      // clamp to edge
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      // no interpolation
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      bt._texture = texture;
    }
    return bt._texture;
  }

  readSync(x: Tensor): TypedArray {
    let ndx = ndarrayOf(x);
    if (!ndx) {
      const bt = backendTensorOf(x);
      if (!bt || !(bt._texture)) {
        throw new Error('No memory data, nor gpu data!');
      }
      const gl = this._glContext;

      // Create a framebuffer backed by the texture
      const framebuffer = gl.createFramebuffer();
      try {
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bt._texture, 0);
        // check if you can read from this type of texture.
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) {
          throw new Error('Can not read from this type of texture.')
        }
        // Read the contents of the framebuffer (data stores the pixel data)
        const rgbaData = new Float32Array(bt._texW * bt._texH * 4);
        gl.readPixels(0, 0, bt._texW, bt._texH, gl.RGBA, gl.FLOAT, rgbaData);
        // Array could be treat as shape [bt._texH, bt._texW, 4]
        const shapeRed = bt._axisY.map(i => bt._shape[i]).concat(bt._axisX.map(i => bt._shape[i]));
        const shapeWithRGBA = shapeRed.push(4);
        const rgbaPickR = ndarray(rgbaData, shapeWithRGBA).pick(...shapeRed.map(i => null), 0);
      
        const rData = new Float32Array(bt._texW * bt._texH);
        bt._array = ndarray(rData, shapeRed);
        nd_ops.assign(bt._array, rgbaPickR);
        ndx = ndarrayOf(x);
      }
      finally {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(framebuffer);
      }
    }
    if (ndx) {
      return ndx.data;
    }
    return null;
  }

  transpose(x: Tensor, perm: number[]): Tensor {
    this.readSync(x);
    const btnda = ndarrayOf(x);
    const trans = btnda.transpose(...perm);
    const y = Tensor.fromBackend(new WebGLTensor(trans));
    y.name = `Tensor${y.id}_transpose_${x.id}`;
    return y;
  }

  matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean):
      Tensor {
    const activeA = (transposeA) ? this.transpose(a, [1, 0]) : a;
    const activeB = (transposeB) ? this.transpose(b, [1, 0]) : b;

    const nda = ndarrayOf(activeA);
    const ndb = ndarrayOf(activeB);
    const c = Tensor.create(null, [nda.shape[0], ndb.shape[1]], nda.dtype);
    const ndc = ndarrayOf(c);
    nd_gemm(ndc, nda, ndb, 1, 0);
    c.name = `Tensor${c.id}_matmul_${a.id}_${b.id}`;
    return c;
  }

  reshape(x: Tensor, newShape: Shape): Tensor {
    // TODO: this is not correct when x do some in place transformation like
    // slice etc
    const numberOfNegs = newShape.reduce(
        (accumulator, value) => accumulator += ((value <= 0) ? 1 : 0), 0);
    const detSize = newShape.reduce(
        (accumulator, value) => accumulator * ((value <= 0) ? 1 : value), 1);
    const oldSize =
        x.shape.reduce((accumulator, value) => accumulator * value, 1);
    const axisSize = oldSize / detSize;
    if (numberOfNegs > 1)
      throw new Error('Too many axises to be flatten in reshape');
    if (numberOfNegs == 1) {
      if (oldSize % detSize != 0)
        throw new Error('Size not matching to flatten');
    }
    const shape = Array(newShape.length);
    for (let i = 0; i < shape.length; ++i) {
      shape[i] = (newShape[i] <= 0) ? axisSize : newShape[i];
    }
    const ta = this.readSync(x);
    const y = Tensor.create(ta, shape, x.dtype);
    y.name = `Tensor${y.id}_reshape_${x.id}`;
    return y;
  }

  add(a: Tensor, b: Tensor): Tensor {
    const backA = ndarrayOf(a);
    const backB = ndarrayOf(b);
    const c = Tensor.create(null, a.shape, a.dtype);
    const backC = ndarrayOf(c);
    if (backA.shape.length == backB.shape.length) {
      // check shape are same
      nd_ops.add(backC, backA, backB);
    } else if (backB.shape.length == 0) {
      nd_ops.adds(backC, backA, backB.data[0]);
    } else {
      // TODO: support better broadcasting...
      const rshape = [];
      for (let i = 0, limit = backA.shape.length - backB.shape.length;
           i < limit; ++i)
        rshape.push(1);
      backB.shape.forEach((v) => rshape.push(v));
      const rtb = this.reshape(b, rshape);
      const ndrtb = ndarrayOf(rtb);
      const bb = broadCastedNdarray(ndrtb, backA.shape);
      nd_ops.add(backC, backA, bb);
    }
    c.name = `Tensor${c.id}_add_${a.id}_${b.id}`;
    return c;
  }
  neg(x: Tensor): Tensor {
    const ndx = ndarrayOf(x);
    const y = Tensor.create(null, x.shape, x.dtype);
    const ndy = ndarrayOf(y);
    nd_ops.neg(ndy, ndx);
    y.name = `Tensor${y.id}_neg_${x.id}`;
    return y;
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const nda = ndarrayOf(a);
    const ndb = ndarrayOf(b);
    const c = Tensor.create(null, a.shape, a.dtype);
    const ndc = ndarrayOf(c);
    nd_ops.multiply(ndc, nda, ndb);
    c.name = `Tensor${c.id}_multiply_${a.id}_${b.id}`;
    return c;
  }

  relu(x: Tensor): Tensor {
    const ndx = ndarrayOf(x);
    const y = Tensor.create(null, x.shape, x.dtype);
    const ndy = ndarrayOf(y);
    nd_ops.maxs(ndy, ndx, 0);
    y.name = `Tensor${y.id}_relu_${x.id}`;
    return y;
  }
  conv2d(
      x: Tensor, filter: Tensor, strides: number|[number, number],
      padding: number[], dataFormat: 'NHWC'|'NCHW',
      dialations: number|[number, number]): Tensor {
    throw new Error('Method not implemented.');
  }
};


const backendName: string = 'Backend_WebGL';
const backendScore: number = 16;
const backendWebGl = new WebGLBackend();
console.info(JSON.stringify(backendWebGl));
if (backendWebGl._isSupported) {
  ENV.registerBackend(backendName, backendWebGl, backendScore);
}
