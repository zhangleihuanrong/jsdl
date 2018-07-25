import { ENV } from "../environments";

import { BackendTensor, DataType, TypedArray, createTypeArrayForShape, Shape } from "../types";
import { Backend } from "../backend";
import { Tensor } from "../tensor";
import * as ndarray from 'ndarray';
import * as nd_ops from 'ndarray-ops';
import * as nd_gemm from 'ndarray-gemm';

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

// TODO: move all ndarray related basic to seperate files.
function rangedArray(s: number) : number[] {
    const arr = new Array(s);
    for (let i = 0; i < s; ++i) arr[i] = i;
    return arr;
}

function broadCastedNdarray(a: ndarray, newShape: number[]) : ndarray {
    a = a.transpose(...rangedArray(a.shape.length));
    const shape : number[] = a.shape;
    const strides : number[] = a.stride;
    shape.forEach((orig, i) => {
        if (orig == 1 && newShape[i] > 1) {
            strides[i] = 0;
        }
    });
    a.shape = newShape;
    return a;
}


function ndarrayOf(t: Tensor) : ndarray {
    return (t.data as WebGLTensor)._array;
}

class WebGLBackend implements Backend {
    _isSupported: boolean = false;
    _canvas: HTMLCanvasElement = null;
    _context: WebGLRenderingContext = null;
    MAX_TEXTURE_SIZE: number;
    MAX_TEXTURE_IMAGE_UNITS: number;
    _refs = { textures: [], buffers: [] };

    constructor() {
        if (typeof window !== 'undefined') {
          this._canvas = document.createElement('canvas');
          if (this._canvas.getContext('webgl2')) {
            this._context = this._canvas.getContext('webgl2') as WebGLRenderingContext;
            this._isSupported = true;
            this._context.getExtension('EXT_color_buffer_float');
            this.MAX_TEXTURE_SIZE = this._context.getParameter(this._context.MAX_TEXTURE_SIZE);
            this.MAX_TEXTURE_IMAGE_UNITS = this._context.getParameter(this._context.MAX_TEXTURE_IMAGE_UNITS);
            this.init();
          } else {
            console.warn('Unable to initialize WebGL2 -- your browser may not support it.');
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
        const y = Tensor.fromBackend(new WebGLTensor(trans));
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
    conv2d(x: Tensor, filter: Tensor, strides: number | [number, number], padding: number[], dataFormat: "NHWC" | "NCHW", dialations: number | [number, number]): Tensor {
        throw new Error("Method not implemented.");
    }
};


const backendName: string = "Backend_WebGL";
const backendScore: number = 16;
const backendWebGl = new WebGLBackend();
console.info(JSON.stringify(backendWebGl));
if (backendWebGl._isSupported) {
    ENV.registerBackend(backendName, backendWebGl, backendScore);
}
