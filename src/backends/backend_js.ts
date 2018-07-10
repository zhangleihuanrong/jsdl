import { StrictTensorLike, TypedArray, getTypedArraySample, toTypedArray } from '../types';
import { Tensor } from '../tensor';

import { Backend } from '../backend';
import { ENV } from '../environments';

import ndarray from 'ndarray';
import gemm from 'ndarray-gemm';
import ops from 'ndarray-ops';

class NdarrayTensor {
    arr: ndarray;
    constructor(arr: ndarray) { 
        this.arr = arr;
    }
};

// function shapePerm(shape: number[], perm: number[]): number[] {
//     const ps: number[] = new Array(shape.length);
//     // TODO: check
//     for (let i = 0; i < shape.length; ++i) {
//         ps[i] = shape[perm[i]];
//     }
//     return ps;
// }

class JsNdarrayBackend implements Backend {
    register(t: Tensor): void {
        if (t.backendTensor == null) {
            const bt : NdarrayTensor = new NdarrayTensor(null);
            t.backendTensor = bt;
        }
    }

    dispose(t: Tensor): void {
        if (t.backendTensor) {
            const bt = t.backendTensor as NdarrayTensor;
            delete bt.arr;
        }
    }

    write(t: Tensor, values: StrictTensorLike): void {
        const bt = t.backendTensor as NdarrayTensor;
        const taSample = getTypedArraySample(t.dtype);
        if (!values) {
            const ta = (t.dtype == 'float32') ? new Float32Array(t.size) :
                       (t.dtype == 'int32') ? new Int32Array(t.size) :
                                              new Uint8Array(t.size);
            bt.arr = ndarray(ta, t.shape);
        }
        else if (values instanceof Array) {
            bt.arr = ndarray(values, t.shape);
        }
        else if (values.constructor == taSample.constructor) {
            bt.arr = ndarray(values, t.shape);
        }
        else {
            // TODO
            throw new Error("Method not implemented.");
        }
    }

    read(t: Tensor): Promise<TypedArray> {
        return new Promise((resolve, reject) => {
            resolve(this.readSync(t));
        });
    }

    readSync(t: Tensor): TypedArray {
        if (t.backendTensor) {
            const bt = t.backendTensor as NdarrayTensor;
            return toTypedArray(bt.arr.data, t.dtype);
        }
        return null;
    }

    transpose(x: Tensor, perm: number[]): Tensor {
        const bt = x.backendTensor as NdarrayTensor;
        const trans = bt.arr.transpose(perm);
        const y = new Tensor(trans.shape, x.dtype, null, new NdarrayTensor(trans));
        return y;
    }

    matMul(a: Tensor, b: Tensor, transposeA: boolean = false, transposeB: boolean = false): Tensor {
        const activeA = (transposeA) ? this.transpose(a, [1, 0]) : a;
        const activeB = (transposeB) ? this.transpose(b, [1, 0]) : b;

        const backA = (activeA.backendTensor as NdarrayTensor).arr;
        const backB = (activeB.backendTensor as NdarrayTensor).arr;
        const C = new Tensor([backA.shape[0], backB.shape[1]], activeB.dtype, null, null);
        const backC = (C.backendTensor as NdarrayTensor).arr;
        
        gemm(backC, a, b, 1, 1);
        return C;
    }

    add(a: Tensor, b: Tensor): Tensor {
        const backA = (a.backendTensor as NdarrayTensor).arr;
        const backB = (b.backendTensor as NdarrayTensor).arr;
        const C = new Tensor(a.shape, a.dtype);
        const backC = (C.backendTensor as NdarrayTensor).arr;
        ops.add(backC, backA, backB);

        return C;
    }

    neg(x: Tensor): Tensor {
        const backX = (x.backendTensor as NdarrayTensor).arr;
        const y = new Tensor(x.shape, x.dtype);
        const backY = (y.backendTensor as NdarrayTensor).arr;
        ops.neg(backY, backX);
        return y;
    }

    multiply(a: Tensor, b: Tensor): Tensor {
        const backA = (a.backendTensor as NdarrayTensor).arr;
        const backB = (b.backendTensor as NdarrayTensor).arr;
        const c = new Tensor(a.shape, a.dtype);
        const backC = (c.backendTensor as NdarrayTensor).arr;
        ops.multiply(backC, backA, backB);
        return c;
    }

    relu(x: Tensor): Tensor {
        const backX = (x.backendTensor as NdarrayTensor).arr;
        const y = new Tensor(x.shape, x.dtype);
        const backY = (y.backendTensor as NdarrayTensor).arr;
        ops.maxs(backY, backX, 0);
        return y;
    }
}

const backendName: string = "JS_ndarray";
const backendScore: number = 2;
const backendJsNdarray = new JsNdarrayBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
