import { StrictTensorLike, TypedArray, getTypedArraySample, toTypedArray, DataType, Shape } from '../types';
import { Tensor } from '../tensor';
import { Backend } from '../backend';
import { ENV } from '../environments';

import * as ndarray from 'ndarray';
import * as nd_gemm from 'ndarray-gemm';
import * as nd_ops from 'ndarray-ops';
import { MPRandGauss } from '../utils/rand';

// function shapePerm(shape: number[], perm: number[]): number[] {
//     const ps: number[] = new Array(shape.length);
//     // TODO: check
//     for (let i = 0; i < shape.length; ++i) {
//         ps[i] = shape[perm[i]];
//     }
//     return ps;
// }
//import { print as tfprint } from '../utils';

function ndarrayOf(t: Tensor) : ndarray {
    return t.data as ndarray;
}

function nd_pad(x: ndarray, paddings: [number, number][], padVal: number = 0) : ndarray {
    //TODO: check parameters
    const padTotal = paddings.reduce((sum, p) => sum + p[0] + p[1], 0);
    if (padTotal == 0) return x;
    const newShape = x.shape.map((len, index) => len + paddings[index][0] + paddings[index][1]);
    const padded = ndarray(x.dtype, newShape);
    const loPoint = paddings.map(beforeAndAfter => beforeAndAfter[0]);
    if (padVal != 0) {
        nd_ops.addseq(padded, padVal);
    }
    nd_ops.assign(padded.lo(loPoint).hi(x.shape), x);
    return padded;
}



class JsNdarrayBackend implements Backend {
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


    //suppose t is from typed array
    randomUniformEq(t: Tensor, a: number, b: number) : void {
        const arr = ndarrayOf(t).data as TypedArray;
        for (let i = 0; i < arr.length; ++i) {
            const v = Math.random() * (b - a) + a;
            arr[i] = v;
        }
    }

    randomNormEq(t: Tensor, mean: number, stdDev: number, seed: number) : void {
        const dtype = t.dtype as 'float32' | 'int32';
        const randGauss = new MPRandGauss(mean, stdDev, dtype, false, seed);
        const arr = ndarrayOf(t).data as TypedArray;
        for (let i = 0; i < arr.length; i++) {
            arr[i] = randGauss.nextValue();
        }
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
        return new Tensor(x.dtype, shape, ta);
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
            for (let i  = 0; i < flatC.shape[1]; i++) {
                const col = flatC.lo(0, i).hi(flatC.shape[0], 1);
                let res = '';
                for (let r = 0; r < flatC.shape[0]; r++) {
                    res += (res.length > 0) ? ', ' : '';
                    res += '' + col.get(r, 0);
                }
                console.log(res);
                nd_ops.addseq(col, flatB.get(i));
            }
        }
        return c;
    }


    //     if (node.opType == "Conv") {
    //         const nameX = node.input[0];
    //         let tensorX = predFlow.tensors[nameX];
    //         assert(tensorX.shape.length == 4); // ensure conv2d
    
    //         tensorX = tf.transpose(tensorX, [0, 2, 3, 1]); // make it NHWC mode
    
    //         const nameW = node.input[1];
    //         let tensorW = predFlow.tensors[nameW];
    //         tensorW = tf.transpose(tensorW, [2, 3, 1, 0]); // make it [H, W, in, out] mode
    
    //         const attrs = node.attribute;
    //         let pads = attrs.find(attr => attr.name == 'pads');
    //         if (pads) {
    //             pads = pads.ints.map(lv => Number(lv));
    //             if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
    //                 const tfpad = [[0, 0], [pads[0], pads[2]], [pads[1], pads[3]], [0, 0]];
    //                 tensorX = tf.pad(tensorX, tfpad, 0);
    //             }
    //         }
    
    //         const strides = attrs.find(attr => attr.name == 'strides').ints.map(lv => Number(lv));
    //         let tensorY = tf.conv2d(tensorX, tensorW, strides, 'valid', 'NHWC');
    //         tensorY = tf.transpose(tensorY, [0, 3, 1, 2]); // make it NCHW
    
    //         const nameY = node.output[0];
    //         predFlow.tensors[nameY] = tensorY;
    //     }
    conv2d(x: Tensor, filter: Tensor, strides: number | [number, number], padding: [number, number, number, number], dataFormat: 'NHWC' | 'NCHW', dilations: number | [number, number] = 1) : Tensor {
        if (!(strides instanceof Array)) strides = [strides as number, strides as number];
        if (!(dilations instanceof Array)) dilations = [dilations as number, dilations as number];
        let ndx = ndarrayOf(x);  // 4d tensor, NHWC or NCHW
        let ndk = ndarrayOf(filter); //[H, W, in, out] or [out, in, H, W]

        if (dataFormat == 'NCHW') {
            ndx = ndx.transpose(0, 2, 3, 1);
            ndk = ndk.transpose(2, 3, 1, 0);
        }

        ndx = nd_pad(ndx, [[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]]);

        // calc output shape
        const batchSize = ndx.shape[0];
        const inputColors = ndx.shape[3];
        const inputRows = ndx.shape[1];
        const inputCols = ndx.shape[2];

        const outChannels = ndk.shape[0];
        const kernelIn = ndk.shape[3];
        const kernelRows = ndx.shape[1];
        const kernelCols = ndx.shape[2];

        const dilateKernelRows = kernelRows + (kernelRows - 1) * (dilations[0] - 1);
        const dilateKernelCols = kernelCols + (kernelCols - 1) * (dilations[1] - 1);

        if (inputColors != kernelIn) throw new Error('intput channel do not match kernnels');
        const outRows = Math.floor((inputRows - dilateKernelRows + strides[0]) / strides[0]);
        const outCols = Math.floor((inputCols - dilateKernelCols + strides[1]) / strides[1]);

        return new Tensor(null, [batchSize, outRows, outCols, outChannels]);
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
