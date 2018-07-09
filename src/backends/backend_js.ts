import { StrictTensorLik, TypedArray, getTypedArraySample, DataType, toTypedArray } from '../types';
import { Tensor } from '../tensor';

import { Backend } from '../backend';
import { ENV } from '../environments';

import ndarray from 'ndarray';
//import ops from 'ndarray-ops';
import gemm from 'ndarray-gemm';

class NdarrayTensor {
    arr: ndarray;
    constructor(arr: ndarray) { 
        this.arr = arr;
    }
};

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

    write(t: Tensor, values: StrictTensorLik): void {
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

    matMul(a: Tensor, b: Tensor, transposeA: boolean, transposeB: boolean): Tensor {
        
        throw new Error("Method not implemented.");
    }
    transpose(x: Tensor, perm: number[]): Tensor {
        throw new Error("Method not implemented.");
    }
    add(a: Tensor, b: Tensor): Tensor {
        throw new Error("Method not implemented.");
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

}

const backendName: string = "JS_ndarray";
const backendScore: number = 2;
const backendJsNdarray = new JsNdarrayBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
