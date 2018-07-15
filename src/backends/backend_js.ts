import { StrictTensorLike, createTypeArrayForShape, TypedArray, getTypedArraySample, DataType, Shape } from '../types';
import { Tensor } from '../tensor';
import { Backend, TensorPrintOptions } from '../backend';
import { ENV } from '../environments';

import * as ndarray from 'ndarray';
import * as nd_gemm from 'ndarray-gemm';
import * as nd_ops from 'ndarray-ops';
import { MPRandGauss } from '../utils/rand';
import { printNdarray, iterateNdarray } from '../utils/ndarray_print';

const sUseRawConv2d = false;

function ndarrayOf(t: Tensor) : ndarray {
    return t.data as ndarray;
}

function nd_pad(x: ndarray, paddings: [number, number][], padVal: number = 0) : ndarray {
    //TODO: check parameters
    const padTotal = paddings.reduce((sum, p) => sum + p[0] + p[1], 0);
    if (padTotal == 0) return x;
    const newShape = x.shape.map((len, index) => len + paddings[index][0] + paddings[index][1]);
    const newSize = newShape.reduce((mul, v) => mul * v, 1);
    const padded = ndarray(new Float32Array(newSize), newShape);
    const loPoint = paddings.map(beforeAndAfter => beforeAndAfter[0]);
    if (padVal != 0) {
        nd_ops.addseq(padded, padVal);
    }
    const sliced = padded.lo(...loPoint).hi(...x.shape);
    nd_ops.assign(sliced, x);
    return padded;
}

const defaultTensorPrintOption = new TensorPrintOptions();

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
            const ta = (dtype == 'int32') ? new Int32Array(size) :
                       (dtype != 'float32') ? new Uint8Array(size) :
                                              new Float32Array(size);
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

    print(x: Tensor, tpo?: TensorPrintOptions) {
        if (x.data) {
            const ndx = ndarrayOf(x);
            tpo = (tpo)? tpo : defaultTensorPrintOption;
            printNdarray(
                ndx, 
                x.name,
                tpo.stringify,
                tpo.excludeLastAxis,
                tpo.excludeHiAxises);
        }
        else {
            console.log(`${x.name} -- TensorId:${x.id} --:  contains no data`)
        }
    }

    randomUniformEq(t: Tensor, a: number, b: number) : void {
        const ndt = ndarrayOf(t);
        iterateNdarray(ndt, (arr: ndarray, loc: number[]) => {
            const v = Math.random() * (b - a) + a;
            arr.set(...loc, v);
        })
    }

    randomNormEq(t: Tensor, mean: number, stdDev: number, seed: number) : void {
        const dtype = t.dtype as 'float32' | 'int32';
        const randGauss = new MPRandGauss(mean, stdDev, dtype, false, seed);
        iterateNdarray(ndarrayOf(t), (ndt:ndarray, loc:number[]) => {
            ndt.set(...loc, randGauss.nextValue());
        })
    }

    transpose(x: Tensor, perm: number[]): Tensor {
        const bt = ndarrayOf(x);
        const trans = bt.transpose(...perm);
        const y = new Tensor(x.dtype, trans.shape, null, trans);
        y.name = `Tensor${y.id}_transpose_${x.id}`;
        return y;
    }

    matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor {
        const activeA = (transposeA) ? this.transpose(a, [1, 0]) : a;
        const activeB = (transposeB) ? this.transpose(b, [1, 0]) : b;

        const nda = ndarrayOf(activeA);
        const ndb = ndarrayOf(activeB);
        const c = new Tensor(nda.dtype, [nda.shape[0], ndb.shape[1]], null, null);
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
        const y = new Tensor(x.dtype, shape, ta);
        y.name = `Tensor${y.id}_reshape_${x.id}`;
        return y;
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
        c.name = `Tensor${c.id}_add_${a.id}_${b.id}`;
        return c;
    }

    conv2d(x: Tensor, filter: Tensor, strides: number | [number, number], padding: number[], dataFormat: 'NHWC' | 'NCHW', dilations: number | [number, number] = 1) : Tensor {
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
        const [batchSize, inputRows, inputCols, inputChannels]  = [ndx.shape[0], ndx.shape[1], ndx.shape[2], ndx.shape[3]];
        const [kernelRows, kernelCols, kernelIn, outChannels] =  [ndk.shape[0], ndk.shape[1], ndk.shape[2], ndk.shape[3]];

        const dilateKernelRows = kernelRows + (kernelRows - 1) * (dilations[0] - 1);
        const dilateKernelCols = kernelCols + (kernelCols - 1) * (dilations[1] - 1);

        if (inputChannels != kernelIn) throw new Error('intput channel do not match kernnels');
        const outRows = Math.floor((inputRows - dilateKernelRows + strides[0]) / strides[0]);
        const outCols = Math.floor((inputCols - dilateKernelCols + strides[1]) / strides[1]);

        const outSize = batchSize * outRows * outCols * outChannels;
        const resultTypedArray = new Float32Array(outSize); 

        if (sUseRawConv2d) {
            let ndr = ndarray(resultTypedArray, [batchSize, outRows, outCols, outChannels])
            for (let b = 0; b < batchSize; ++b) {
                for (let d2 = 0; d2 < outChannels; ++d2) {
                    for (let yR = 0; yR < outRows; ++yR) {
                        const xRCorner = yR * strides[0];
                        for (let yC = 0; yC < outCols; ++yC) {
                            const xCCorner = yC * strides[1];
    
                            let dotProd = 0;
                            for (let wR = 0; wR < kernelRows; wR++) {
                                const xR = xRCorner + wR * dilations[0];
    
                                if (xR < 0 || xR >= inputRows) {
                                    continue;
                                }
    
                                for (let wC = 0; wC < kernelCols; wC++) {
                                    const xC = xCCorner + wC * dilations[1];
    
                                    if (xC < 0 || xC >= inputCols) {
                                        continue;
                                    }
    
                                    for (let d1 = 0; d1 < inputChannels; ++d1) {
                                        const pixel = ndx.get(b, xR, xC, d1);
                                        const weight = ndk.get(wR, wC, d1, d2);
                                        dotProd += pixel * weight;
                                    }
                                }
                            }
                            ndr.set(b, yR, yC, d2, dotProd);
                        }
                    }
                }
            }
        }
        else {
            // prepare the col image
            const numberOfPatches = batchSize * outRows * outCols;
            const sizeOfPatch = kernelRows * kernelCols * inputChannels;
            const patch = ndarray(new Float32Array(sizeOfPatch), [kernelRows, kernelCols, inputChannels]);
            let colImage : ndarray = null;
            if (dilateKernelRows == 1 && dilateKernelCols == 1 && strides[0] == 1 && strides[1] == 1) {
                colImage = ndarray(ndx.data, [numberOfPatches, sizeOfPatch]);  // infact a reshape
            }
            else {
                colImage = ndarray(new Float32Array(numberOfPatches * sizeOfPatch), [numberOfPatches, sizeOfPatch]);
                let offset = 0;
                for (let b = 0; b < batchSize; ++b) {
                    for (let r = 0, rMax = inputRows - dilateKernelRows + 1; r < rMax; r += dilateKernelRows) {
                        for (let c = 0, cMax = inputCols - dilateKernelCols + 1; c < cMax; c += dilateKernelCols) {
                            let patchView = ndx.pick(b, null, null, null);
                            patchView = patchView.hi(r + dilateKernelRows, c + dilateKernelCols, inputChannels);
                            patchView = patchView.lo(r, c, 0);
                            patchView = patchView.step(dilations[0], dilations[1], 1);
                            nd_ops.assign(patch, patchView);
                            colImage.data.set(patch.data, offset);
                            offset += sizeOfPatch;
                        }
                    }
                }
            }

            // prepare the col image
            const ndf = ndarray(new Float32Array(sizeOfPatch * outChannels), [sizeOfPatch, outChannels]);
            for (let ch = 0; ch < outChannels; ++ch) {
                nd_ops.assign(patch, ndk.pick(null, null, null, ch));
                const reshaped = ndarray(patch.data, [sizeOfPatch]);
                nd_ops.assign(ndf.pick(null, ch), reshaped);
            }
            
            let ndr = ndarray(resultTypedArray, [numberOfPatches, outChannels]);
            nd_gemm(ndr, colImage, ndf, 1, 1);
        }

        const r = new Tensor('float32', [batchSize, outRows, outCols, outChannels], resultTypedArray);
        r.name = `Tensor${r.id}_conv2D_${x.id}`;
        return r;
    }


    neg(x: Tensor): Tensor {
        const ndx = ndarrayOf(x);
        const y = new Tensor(x.dtype, x.shape);
        const ndy = ndarrayOf(y);
        nd_ops.neg(ndy, ndx);
        y.name = `Tensor${y.id}_neg_${x.id}`;
        return y;
    }

    multiply(a: Tensor, b: Tensor): Tensor {
        const nda = ndarrayOf(a);
        const ndb = ndarrayOf(b);
        const c = new Tensor(a.dtype, a.shape);
        const ndc = ndarrayOf(c);
        nd_ops.multiply(ndc, nda, ndb);
        c.name = `Tensor${c.id}_multiply_${a.id}_${b.id}`;
        return c;
    }

    relu(x: Tensor): Tensor {
        const ndx = ndarrayOf(x);
        const y = new Tensor(x.dtype, x.shape);
        const ndy = ndarrayOf(y);
        nd_ops.maxs(ndy, ndx, 0);
        y.name = `Tensor${y.id}_relu_${x.id}`;
        return y;
    }
}

const backendName: string = "JS_ndarray";
const backendScore: number = 2;
const backendJsNdarray = new JsNdarrayBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
