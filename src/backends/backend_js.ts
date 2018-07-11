import { StrictTensorLike, TypedArray, getTypedArraySample, toTypedArray, DataType, Shape } from '../types';
import { Tensor } from '../tensor';

import { Backend } from '../backend';
import { ENV } from '../environments';

import ndarray from 'ndarray';
import gemm from 'ndarray-gemm';
import ops from 'ndarray-ops';

// function shapePerm(shape: number[], perm: number[]): number[] {
//     const ps: number[] = new Array(shape.length);
//     // TODO: check
//     for (let i = 0; i < shape.length; ++i) {
//         ps[i] = shape[perm[i]];
//     }
//     return ps;
// }

class JsNdarrayBackend implements Backend {
    ndarrayOf(t: Tensor) : ndarray {
        return t.data as ndarray;
    }

    tensorShape(t: Tensor): number[] {
        return (t.data as ndarray).shape;
    }

    tensorDtype(t: Tensor): DataType {
        return (t.data as ndarray).dtype as DataType;
    }

    tensorSize(t: Tensor): number {
        return (t.data as ndarray).size;
    }

    wrap(t: Tensor, dtype: DataType, shape: Shape, backendTensor: object): void {
        // TODO: check type, shape, etc if needed
        t.data = backendTensor;
    }

    make(t: Tensor, dtype: DataType, shape: Shape, values: StrictTensorLike): void {
        const taSample = getTypedArraySample(t.dtype);
        if (!values) {
            const size = shape.reduce((a, b) => a*b, 1);
            const ta = (dtype == 'float32') ? new Float32Array(size) :
                       (dtype == 'int32') ? new Int32Array(size) :
                                              new Uint8Array(size);
            t.data = ndarray(ta, t.shape);
        }
        else if (values instanceof Array) {
            t.data = ndarray(values, shape);
        }
        else if (values.constructor == taSample.constructor) {
            t.data = ndarray(values, shape);
        }
        else {
            // TODO
            throw new Error("Method not implemented.");
        }
    }

    free(t: Tensor): void {
        delete t.data;
    }

    // read(t: Tensor): Promise<TypedArray> {
    //     return new Promise((resolve, reject) => {
    //         resolve(this.readSync(t));
    //     });
    // }

    readSync(t: Tensor): TypedArray {
        if (t.data) {
            return toTypedArray((t.data as ndarray).data, t.dtype);
        }
        return null;
    }

    transpose(x: Tensor, perm: number[]): Tensor {
        const bt = x.data as ndarray;
        const trans = bt.transpose(perm);
        const y = new Tensor(x.dtype, trans.shape, null, new ndarray(trans));
        return y;
    }

    matMul(a: Tensor, b: Tensor, transposeA: boolean = false, transposeB: boolean = false): Tensor {
        const activeA = (transposeA) ? this.transpose(a, [1, 0]) : a;
        const activeB = (transposeB) ? this.transpose(b, [1, 0]) : b;

        const backA = (activeA.data as ndarray);
        const backB = (activeB.data as ndarray);
        const C = new Tensor(activeB.dtype, [backA.shape[0], backB.shape[1]], null, null);
        const backC = (C.data as ndarray).arr;
        gemm(backC, a, b, 1, 1);
        return C;
    }

    add(a: Tensor, b: Tensor): Tensor {
        const backA = (a.data as ndarray);
        const backB = (b.data as ndarray);
        const c = new Tensor(a.dtype, a.shape);
        const backC = (c.data as ndarray);
        ops.add(backC, backA, backB);
        return c;
    }

    neg(x: Tensor): Tensor {
        const backX = (x.data as ndarray);
        const y = new Tensor(x.dtype, x.shape);
        const backY = (y.data as ndarray);
        ops.neg(backY, backX);
        return y;
    }

    multiply(a: Tensor, b: Tensor): Tensor {
        const backA = (a.data as ndarray);
        const backB = (b.data as ndarray);
        const c = new Tensor(a.dtype, a.shape);
        const backC = (c.data as ndarray);
        ops.multiply(backC, backA, backB);
        return c;
    }

    relu(x: Tensor): Tensor {
        const backX = (x.data as ndarray);
        const y = new Tensor(x.dtype, x.shape);
        const backY = (y.data as ndarray);
        ops.maxs(backY, backX, 0);
        return y;
    }
}

const backendName: string = "JS_ndarray";
const backendScore: number = 2;
const backendJsNdarray = new JsNdarrayBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
