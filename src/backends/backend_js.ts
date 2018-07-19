import { createTypeArrayForShape, TypedArray, DataType, Shape, BackendTensor } from '../types';
import { Tensor } from '../tensor';
import { Backend } from '../backend';
import { ENV } from '../environments';
import * as ndarray from 'ndarray';
import * as nd_gemm from 'ndarray-gemm';
import * as nd_ops from 'ndarray-ops';

class NdarrayTensor implements BackendTensor {
    _array: ndarray;

    constructor(nda: ndarray) {
        this._array = nda;
    }

    shape(): number[] {
        return this._array.shape;;
    }
    dtype(): DataType {
        // TODO: not fully compatible
        return this._array.dtype as DataType;;
    }
    size(): number {
        return this._array.size;
    }
}

function ndarrayOf(t: Tensor) : ndarray {
    return (t.data as NdarrayTensor)._array;
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

class JsNdarrayBackend implements Backend {
    wrap(t: Tensor, backendTensor: BackendTensor): void {
        t.data = backendTensor;
    }

    make(t: Tensor, dtype: DataType, shape: Shape, values: TypedArray): void {
        t.data = new NdarrayTensor(ndarray(values, shape));
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

    // TODO: support better broadcasting...
    add(a: Tensor, b: Tensor): Tensor {
        const backA = ndarrayOf(a);
        const backB = ndarrayOf(b);
        const c = Tensor.create(null, a.shape, a.dtype);
        const backC = ndarrayOf(c);
        if (backA.shape.length == backB.shape.length) {
            // check shape are same
            nd_ops.add(backC, backA, backB);
        } else {
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
        const [batchSize, inputRows, inputCols, inputChannels] = [ndx.shape[0], ndx.shape[1], ndx.shape[2], ndx.shape[3]];
        const [kernelRows, kernelCols, kernelIn, outChannels] =  [ndk.shape[0], ndk.shape[1], ndk.shape[2], ndk.shape[3]];

        const dilateKernelRows = kernelRows + (kernelRows - 1) * (dilations[0] - 1);
        const dilateKernelCols = kernelCols + (kernelCols - 1) * (dilations[1] - 1);

        if (inputChannels != kernelIn) throw new Error('intput channel do not match kernnels');
        const outRows = Math.floor((inputRows - dilateKernelRows + strides[0]) / strides[0]);
        const outCols = Math.floor((inputCols - dilateKernelCols + strides[1]) / strides[1]);

        const outSize = batchSize * outRows * outCols * outChannels;
        const resultTypedArray = new Float32Array(outSize); 

        const sizeOfPatch = kernelRows * kernelCols * inputChannels;
        const patch = ndarray(new Float32Array(sizeOfPatch), [kernelRows, kernelCols, inputChannels]);
        const ndf = ndarray(new Float32Array(sizeOfPatch * outChannels), [sizeOfPatch, outChannels]);
        for (let yC = 0; yC < outChannels; ++yC) {
            nd_ops.assign(patch, ndk.pick(null, null, null, yC));
            const reshaped = ndarray(patch.data, [sizeOfPatch]);
            nd_ops.assign(ndf.pick(null, yC), reshaped);
        }

        const ndr = ndarray(resultTypedArray, [batchSize, outRows, outCols, outChannels]);
        const pixelResult = ndarray(new Float32Array(outChannels), [1, outChannels]);
        let offset = 0;
        for (let b = 0; b < batchSize; ++b) {
            let singleImage = ndx.pick(b, null, null, null);
            for (let yH = 0; yH < outRows; ++yH) {
                let xH = yH * strides[0];
                for (let yW = 0; yW < outCols; ++yW) {
                    let xW = yW * strides[1];

                    let patchView = singleImage
                            .hi(xH + dilateKernelRows, xW + dilateKernelCols, inputChannels)
                            .lo(xH, xW, 0)
                            .step(dilations[0], dilations[1], 1);
                    nd_ops.assign(patch, patchView);
                    const patchInRow:ndarray = ndarray(patch.data, [1, sizeOfPatch]);

                    nd_gemm(pixelResult, patchInRow, ndf);
                    ndr.data.set(pixelResult.data, offset);
                    offset += outChannels;
                }
            }
        }

        const r = Tensor.create(resultTypedArray, [batchSize, outRows, outCols, outChannels]);
        r.name = `Tensor${r.id}_conv2D_${x.id}`;
        return r;
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
}

const backendName: string = "JS_ndarray";
const backendScore: number = 2;
const backendJsNdarray = new JsNdarrayBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
