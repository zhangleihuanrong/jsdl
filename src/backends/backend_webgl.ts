import {Backend} from '../backend';
import {ENV} from '../environments';
import {NDView as NdArray} from '../NdView/ndview';
import {Tensor} from '../tensor';
import {BackendTensor, DataType, Shape, TypedArray} from '../types';
import {assert as ASSERT} from '../utils/gadget';

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
  // basic info, TODO: combine with _array: ndarray
  _dtype: DataType;

  _array: NdArray;

  // Other fields will be used by webgl
  _texture: WebGLTexture;  // 2D Texture
  _texW: number;
  _texH: number;
  _axisX: number[];
  _axisY: number[];


  constructor(nda: NdArray, dtype: DataType = 'float32') {
    this._dtype = dtype;
    this._array = nda;
  }

  calulateTexture2DShape(MAX_TEXTURE_SIZE: number): boolean {
    const shape = this.shape();
    if (shape.length == 1) {
      this._texW = shape[0];
      this._axisX = [0];
      this._texH = 1;
      this._axisY = [];
    } else if (shape.length == 2) {
      this._texW = shape[1];
      this._axisX = [1];
      this._texH = shape[0];
      this._axisY = [0];
    } else {
      // TODO: current is hacky
      let m = 0;
      while (m < shape.length && shape[m] == 1) --m;

      this._texW = 1;
      this._axisX = [];
      let i = shape.length - 1;
      while (i > 0 && (i >= m || this._texW == 1) &&
             this._texW * shape[i] <= MAX_TEXTURE_SIZE) {
        this._texW *= shape[i];
        this._axisX.unshift(i);
        --i;
      }
      this._texH = 1;
      this._axisY = [];
      for (; i >= 0; --i) {
        this._texH *= shape[i];
        this._axisY.unshift(i);
      }
    }
    return (
        this._texW <= MAX_TEXTURE_SIZE && this._texH <= MAX_TEXTURE_SIZE &&
        this._axisX.length <= 4 && this._axisY.length <= 4);
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


function NdArrayOf(t: Tensor): NdArray {
  return (t.data) ? (t.data as WebGLTensor)._array : null;
}


function backendTensorOf(t: Tensor): WebGLTensor {
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
      this._glContext =
          this._canvas.getContext('webgl2') as WebGL2RenderingContext;
      if (this._glContext) {
        const gl = this._glContext;
        this._isSupported = true;
        gl.getExtension('EXT_color_buffer_float');
        this.MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
        this.MAX_TEXTURE_IMAGE_UNITS =
            gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
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
   * Store reference to WebGL texture or buffer on class instance, useful for
   * when we want to delete later
   */
  storeRef(texOrBuf: WebGLTexture|WebGLBuffer) {
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
    const gl = this._glContext;
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

    const vs = new Float32Array(
        [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]);
    gl.bufferData(gl.ARRAY_BUFFER, vs, gl.STATIC_DRAW);
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(position);

    const texcoord = gl.getAttribLocation(program, 'texcoord');
    const texcoordVertexObj = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texcoordVertexObj);
    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        gl.STATIC_DRAW);
    gl.vertexAttribPointer(texcoord, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(texcoord);
    this.storeRef(texcoordVertexObj);

    const indicesVertexObj = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indicesVertexObj);
    gl.bufferData(
        gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]),
        gl.STATIC_DRAW);
    this.storeRef(indicesVertexObj);
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


  bindOutputTexture(outputTexture: WebGLTexture, shape: number[], framebuffer?: WebGLFramebuffer) {
    const gl = this._glContext;
    gl.viewport(0, 0, shape[1], shape[0]);
    const activeFramebuffer = framebuffer || gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, activeFramebuffer);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);
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
      const tt = (bt._dtype == 'int32') ?
          gl.INT :
          ((bt._dtype == 'bool') ? gl.BYTE : gl.FLOAT);

      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.R32F, bt._texW, bt._texH, 0, gl.RED, tt,
          bt._array.data as TypedArray, 0);

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

  getCpuMemArray(x: Tensor): NdArray{
    ASSERT(backendTensorOf(x) != null, "No backend tensor found");

    let ndx = NdArrayOf(x);
    if (!ndx) {
      const bt = backendTensorOf(x);
      if (!(bt._texture)) {
        throw new Error('No memory data, nor gpu data!');
      }
      const gl = this._glContext;

      // Create a framebuffer backed by the texture
      const framebuffer = gl.createFramebuffer();
      try {
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bt._texture,
            0);
        // check if you can read from this type of texture.
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !=
            gl.FRAMEBUFFER_COMPLETE) {
          throw new Error('Can not read from this type of texture.')
        }
        // Read the contents of the framebuffer (data stores the pixel data)
        const rgbaData = new Float32Array(bt._texW * bt._texH * 4);
        gl.readPixels(0, 0, bt._texW, bt._texH, gl.RGBA, gl.FLOAT, rgbaData);

        // Array could be treat as shape [bt._texH, bt._texW, 4]
        const shapeRed = bt._axisY.map(i => bt.shape()[i])
                              .concat(bt._axisX.map(i => bt.shape()[i]));
        const shapeWithRGBA = shapeRed.slice(0);
        shapeWithRGBA.push(4);
        ndx = new NdArray(rgbaData, shapeWithRGBA).pick([
          ...shapeRed.map(i => null), 0
        ]);
      } finally {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(framebuffer);
      }
    }
    return ndx;
  }

  readSync(x: Tensor): TypedArray {
    let ndx = this.getCpuMemArray(x);
    if (ndx) {
      return ndx.rebuild().data as TypedArray;
    }
    return null;
  }

  transpose(x: Tensor, perm: number[]): Tensor {
    const bt = backendTensorOf(x);
    if (!bt._texture) this.create2DGLTexture(bt);

    const nda = this.getCpuMemArray(x).transpose(perm);
    return Tensor.fromBackend(new WebGLTensor(nda, x.dtype));
  }

  matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean):
      Tensor {
    throw new Error('Not implemented');
  }

  reshape(x: Tensor, newShape: Shape): Tensor {
    const nda = this.getCpuMemArray(x).reshape(newShape);
    return Tensor.fromBackend(new WebGLTensor(nda, x.dtype));
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
