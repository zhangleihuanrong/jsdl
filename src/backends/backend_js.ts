import { StrictTensorLike, TypedArray, getTypedArraySample, toTypedArray, DataType, Shape } from '../types';
import { Tensor } from '../tensor';
import { Backend } from '../backend';
import { ENV } from '../environments';

import * as ndarray from 'ndarray';
import * as nd_gemm from 'ndarray-gemm';
import * as nd_ops from 'ndarray-ops';

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
        const arrayData = (t.data as ndarray).data;
        if (arrayData instanceof Int32Array) {
            return 'int32';
        }
        else if (arrayData instanceof Uint8Array) {
            return 'bool'
        } 
        else {
            return 'float32';
        }
    }

    tensorSize(t: Tensor): number {
        return (t.data as ndarray).size;
    }

    wrap(t: Tensor, dtype: DataType, shape: Shape, backendTensor: object): void {
        // TODO: check type, shape, etc if needed
        t.data = backendTensor;
    }

    make(t: Tensor, dtype: DataType, shape: Shape, values: StrictTensorLike): void {
        const taSample = getTypedArraySample(dtype);
        if (!values) {
            const size = shape.reduce((a, b) => a*b, 1);
            const ta = (dtype == 'float32') ? new Float32Array(size) :
                       (dtype == 'int32') ? new Int32Array(size) :
                                              new Uint8Array(size);
            const arr = ndarray(ta, shape);
            t.data = arr;
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

    matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor {
        const activeA = (transposeA) ? this.transpose(a, [1, 0]) : a;
        const activeB = (transposeB) ? this.transpose(b, [1, 0]) : b;

        const backA = (activeA.data as ndarray);
        const backB = (activeB.data as ndarray);
        const c = new Tensor(activeB.dtype, [backA.shape[0], backB.shape[1]], null, null);
        const backC = (c.data as ndarray);
        nd_gemm(backC, backA, backB, 1, 0);
        return c;
    }

    reshape(x: Tensor, newShape: Shape) : Tensor {
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
        return new Tensor(this.tensorDtype(x), shape, ta);
    }

    add(a: Tensor, b: Tensor): Tensor {
        const backA = (a.data as ndarray);
        const backB = (b.data as ndarray);
        const c = new Tensor(a.dtype, a.shape);
        const backC = (c.data as ndarray);
        if (backA.shape.length == backB.shape.length) {
            nd_ops.add(backC, backA, backB);
        } else { // only support one dimension more
            nd_ops.assign(backC, backA);
            const flatB = (this.reshape(b, [-1])).data as ndarray;;
            const flatC = (this.reshape(c, [backC.shape[0], -1])).data as ndarray;
            for (let i  = 0; i < backC.shape[0]; i++) {
                const row = flatC.low(i, 0).hi(i, flatB.size);
                nd_ops.addeq(row, flatB);
            }
        }
        return c;
    }

    neg(x: Tensor): Tensor {
        const backX = (x.data as ndarray);
        const y = new Tensor(x.dtype, x.shape);
        const backY = (y.data as ndarray);
        nd_ops.neg(backY, backX);
        return y;
    }

    multiply(a: Tensor, b: Tensor): Tensor {
        const backA = (a.data as ndarray);
        const backB = (b.data as ndarray);
        const c = new Tensor(a.dtype, a.shape);
        const backC = (c.data as ndarray);
        nd_ops.multiply(backC, backA, backB);
        return c;
    }

    relu(x: Tensor): Tensor {
        const backX = (x.data as ndarray);
        const y = new Tensor(x.dtype, x.shape);
        const backY = (y.data as ndarray);
        nd_ops.maxs(backY, backX, 0);
        return y;
    }
}

const backendName: string = "JS_ndarray";
const backendScore: number = 2;
const backendJsNdarray = new JsNdarrayBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
