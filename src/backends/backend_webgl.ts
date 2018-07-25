import { BackendTensor, DataType, TypedArray } from "../types";
import * as ndarray from 'ndarray';
import { Backend } from "../backend";
import { Tensor } from "../tensor";
import { ENV } from "../environments";

class WebGLTensor implements BackendTensor {
    _array: ndarray;
    // Other field will be used by webgl.
    _id: object;

    constructor(nda: ndarray) {
        this._array = nda;
        this._id = null;
    }
    shape(): number[] {
        return this._array.shape;
    }   
    dtype(): DataType {
        // TODO: not fully compatible
        return this._array.dtype as DataType;;
    }
    size(): number {
        return this._array.size;
    }
};


class WebGLBackend implements Backend {
    _isSupported: boolean = false;
    _canvas: HTMLCanvasElement = null;
    _context: WebGLRenderingContext = null;
    MAX_TEXTURE_SIZE: number;
    MAX_TEXTURE_IMAGE_UNITS: number;
    _refs = { textures: [], buffers: [] };

    constructor() {
        this._isSupported = false

        if (typeof window !== 'undefined') {
          this._canvas = document.createElement('canvas')
          this._context = this._canvas.getContext('webgl2') as WebGLRenderingContext;
    
          const gl = this._context;
          if (gl) {
            this._isSupported = true
            gl.getExtension('EXT_color_buffer_float')
            this.MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE)
            this.MAX_TEXTURE_IMAGE_UNITS = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
            this.init()
          } else {
            console.log('Unable to initialize WebGL2 -- your browser may not support it.')
          }
        }
    }
    init() {

    }
    wrap(t: Tensor, backendTensor: BackendTensor): void {
        t.data = backendTensor;
    }

    make(t: Tensor, dtype: DataType, shape: number[], values: TypedArray): void {
        if (dtype != 'float32') throw ('not supported yet');
        t.data = new WebGLTensor(ndarray(values, shape));
    }

    free(t: Tensor): void {
        // ???
        delete t.data;
    }

    readSync(x: Tensor): TypedArray {
        if (x.data) {
            const ndx = ndarrayOf(x);
            const shape = ndx.shape;
            const dtype = x.dtype;
            const ta = createTypeArrayForShape(dtype, shape);
            const r = ndarray(ta, shape);
            nd_ops.assign(r, ndx);
            return ta;
        }
        return null;
    }

    transpose(x: Tensor, perm: number[]): Tensor {
        const bt = ndarrayOf(x);
        const trans = bt.transpose(...perm);
        const y = Tensor.fromBackend(new NdarrayTensor(trans));
        y.name = `Tensor${y.id}_transpose_${x.id}`;
        return y;
    }

    matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor {
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

    reshape(x: Tensor, newShape: Shape) : Tensor {
        // TODO: this is not correct when x do some in place transformation like slice etc
        const numberOfNegs = newShape.reduce((accumulator, value) => accumulator += ((value <= 0)? 1 : 0), 0);
        const detSize = newShape.reduce((accumulator, value) => accumulator * ((value <= 0)? 1 : value), 1);
        const oldSize = x.shape.reduce((accumulator, value) => accumulator * value, 1);
        const axisSize = oldSize / detSize;
        if (numberOfNegs > 1) throw new Error('Too many axises to be flatten in reshape');
        if (numberOfNegs == 1) {
            if (oldSize % detSize != 0) throw new Error('Size not matching to flatten');
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
        } 
        else if (backB.shape.length == 0) {
            nd_ops.adds(backC, backA, backB.data[0]);
        }
        else {
            // TODO: support better broadcasting...
            const rshape = [];
            for (let i = 0, limit = backA.shape.length - backB.shape.length; i < limit; ++i) rshape.push(1);
            backB.shape.forEach((v) => rshape.push(v));
            const rtb = this.reshape(b, rshape);
            const ndrtb = ndarrayOf(rtb);
            const bb = broadCastedNdarray(ndrtb, backA.shape);
            nd_ops.add(backC, backA, bb);
        }
        c.name = `Tensor${c.id}_add_${a.id}_${b.id}`;
        return c;
    }
    neg(a: Tensor): Tensor {
        throw new Error("Method not implemented.");
    }
    multiply(a: Tensor, b: Tensor): Tensor {
        throw new Error("Method not implemented.");
    }
    relu(x: Tensor): Tensor {
        throw new Error("Method not implemented.");
    }
    conv2d(x: Tensor, filter: Tensor, strides: number | [number, number], padding: number[], dataFormat: "NHWC" | "NCHW", dialations: number | [number, number]): Tensor {
        throw new Error("Method not implemented.");
    }
};


const backendName: string = "WebGL_Backend";
const backendScore: number = 2;
const backendJsNdarray = new WebGLBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
