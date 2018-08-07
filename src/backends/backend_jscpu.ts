import { ENV } from '../environments';
import { createTypeArrayForShape, TypedArray, DataType, Shape, BackendTensor } from '../types';
import { Tensor } from '../tensor';
import { Backend } from '../backend';
import { assert as ASSERT } from '../utils/gadget';

import { NDView as NdArray } from '../NdView/ndview';
import { canBroadcastTo, getUnsqueezeAxisForBroadcast, getUnsqueezedShapeForBroadcast, getBroadcastRepeats } from '../utils/shapeTools';
import * as nd from 'ndarray';
import * as nd_gemm from 'ndarray-gemm';
import * as nd_ops from 'ndarray-ops';

class NdArrayTensor implements BackendTensor {
    _dtype: DataType;
    _array: NdArray;

    constructor(nda: NdArray, dtype: DataType) {
        this._dtype = dtype;
        this._array = nda;
    }

    get dtype(): DataType {
        return this._dtype;
    }

    get shape(): number[] {
        return this._array.shape;
    }

    get size(): number {
        return this._array.size;
    }
}

function backendTensorOf(t: Tensor): NdArrayTensor {
    return t.data as NdArrayTensor;
}

function NdArrayOf(t: Tensor): NdArray {
    return (t.data as NdArrayTensor)._array;
}

class JsNdarrayBackend implements Backend {
    wrap(t: Tensor, backendTensor: BackendTensor): void {
        t.data = backendTensor;
    }

    make(t: Tensor, dtype: DataType, shape: Shape, values: TypedArray): void {
        t.data = new NdArrayTensor(new NdArray(values, shape), dtype);
    }

    // ???
    free(t: Tensor): void {
        delete t.data;
    }

    read(x: Tensor): TypedArray {
        const bt = backendTensorOf(x);
        if (bt._array) {
            return bt._array.rebuild().data as TypedArray;
        }
        return null;
    }

    transpose(x: Tensor, perm?: number[]): Tensor {
        const bt = backendTensorOf(x);
        const nda = bt._array.transpose(perm);
        return Tensor.fromBackend(new NdArrayTensor(nda, bt._dtype));
    }

    reshape(x: Tensor, newShape: Shape): Tensor {
        const bt = backendTensorOf(x);
        const nda = bt._array.reshape(newShape);
        return Tensor.fromBackend(new NdArrayTensor(nda, bt._dtype));
    }

    matMul(a: Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor {
        let A = NdArrayOf(a);
        let B = NdArrayOf(b);

        if (transposeA) A = A.transpose();
        if (transposeB) B = B.transpose();

        if (A.shape.length == 1) A = A.expandDim(0);
        if (B.shape.length == 1) B = B.expandDim(1);

        const shapeAMul = A.shape.slice(A.shape.length - 2);
        const shapeBMul = B.shape.slice(B.shape.length - 2);
        ASSERT(shapeAMul[1] == shapeBMul[0], `shape[${shapeAMul}] can not matMul with shape[${shapeBMul}]`);
        const commonDim = shapeAMul[1];

        const shapeAPrefix = A.shape.slice(0, A.shape.length - 2);
        const shapeBPrefix = B.shape.slice(0, B.shape.length - 2);
        if (shapeAPrefix.length > 0) {
            ASSERT(canBroadcastTo(shapeAPrefix, shapeBPrefix), "Can not broadcast b to a");
            const unsq = getUnsqueezeAxisForBroadcast(shapeAPrefix, shapeBPrefix);
            if (unsq) B = B.unsqueeze(unsq);
            const shapeBExpanded = getUnsqueezedShapeForBroadcast(shapeAPrefix, shapeBPrefix);
            const repeats = getBroadcastRepeats(shapeAPrefix, shapeBExpanded);
            if (repeats) B = B.tile(repeats.concat(1, 1));
        }

        const shapeC = shapeAPrefix.concat(shapeAMul[0], shapeBMul[1]);
        const rankC = shapeC.length;
        const ta = createTypeArrayForShape(a.dtype, shapeC);
        const C = new NdArray(ta, shapeC);

        if (A.isCoreOnly() && B.isCoreOnly() && A.shape.length == 2) {
            const ndA = nd(A.data, A.shape);
            const ndB = nd(B.data, B.shape);
            const ndC = nd(C.data, C.shape);
            nd_gemm(ndC, ndA, ndB);
            return Tensor.fromBackend(new NdArrayTensor(C, a.dtype));
        }

        const codeLines: string[] = [];
        //codeLines.push(`const matMulFunc = function(C, A, B) {`);
        let indent = '    ';
        const gatherAsnippet = A.generateGatherDefinedCode('gatherA_', indent);
        if (gatherAsnippet) codeLines.push(gatherAsnippet);
        const gatherBsnippet = B.generateGatherDefinedCode('gatherB_', indent);
        if (gatherBsnippet) codeLines.push(gatherBsnippet);
        for (let h = 0; h < rankC; ++h) {
            codeLines.push(`${indent}for (let i${h} = 0; i${h} < ${shapeC[h]}; ++i${h}) {`);
            indent = `${indent}    `;
            if (h != rankC - 1) {
                codeLines.push(A.generateCoreIndexOnAxisCode(h, `i${h}`, `ai${h}`, `gatherA_${h}`, `${indent}`));
            }
            if (h != rankC - 2) {
                codeLines.push(B.generateCoreIndexOnAxisCode(h, `i${h}`, `bi${h}`, `gatherB_${h}`, `${indent}`));
            }

            if (h == 0) {
                codeLines.push(`${indent}let offsetC${h} = ${C.coreOffset} + i${h} * ${C.coreStride[h]};`);
            }
            else {
                codeLines.push(`${indent}let offsetC${h} = offsetC${h - 1}  + i${h} * ${C.coreStride[h]};`);
            }
            codeLines.push(``);
        }

        codeLines.push(`${indent}let sum = 0;`);
        codeLines.push(`${indent}for (let k = 0; k < ${commonDim}; ++k) {`);
        indent = `${indent}    `;
        codeLines.push(A.generateCoreIndexOnAxisCode(rankC - 1, `k`, `ak`, `gatherA_${rankC - 1}`, `${indent}`));
        const axisNamesA = A.shape.map((v, i) => (i == rankC - 1) ? `ak` : `ai${i}`);
        codeLines.push(`${indent}let aPosition = ${A.generateCalcPosition(axisNamesA)};`)

        codeLines.push(B.generateCoreIndexOnAxisCode(rankC - 2, `k`, `bk`, `gatherA_${rankC - 2}`, `${indent}`));
        const axisNamesB = B.shape.map((v, i) => (i == rankC - 2) ? `bk` : `bi${i}`);
        codeLines.push(`${indent}let bPosition = ${B.generateCalcPosition(axisNamesB)};`)

        codeLines.push(`${indent}sum += A[aPosition] * B[bPosition];`);
        indent = indent.slice(0, indent.length - 4);
        codeLines.push(`${indent}}`);
        codeLines.push(`${indent}C[offsetC${rankC - 1}] = sum;`);

        for (let h = 0; h < rankC; ++h) {
            indent = indent.slice(0, indent.length - 4);
            codeLines.push(`${indent}}`);
        }

        const matMulCode = codeLines.join('\n');
        const matMulFunc = new Function('C', 'A', 'B', matMulCode);
        matMulFunc(C.data, A.data, B.data);
        return Tensor.fromBackend(new NdArrayTensor(C, a.dtype));
    }


    add(a: Tensor, b: Tensor): Tensor {
        // const backA = NdArrayOf(a);
        // const backB = NdArrayOf(b);
        // const c = Tensor.create(null, a.shape, a.dtype);
        // const backC = NdArrayOf(c);
        // if (backA.shape.length == backB.shape.length) {
        //     // check shape are same
        //     nd_ops.add(backC, backA, backB);
        // } 
        // else if (backB.shape.length == 0) {
        //     nd_ops.adds(backC, backA, backB.data[0]);
        // }
        // else {
        //     // TODO: support better broadcasting...
        //     const rshape = [];
        //     for (let i = 0, limit = backA.shape.length - backB.shape.length; i < limit; ++i) rshape.push(1);
        //     backB.shape.forEach((v) => rshape.push(v));
        //     const rtb = this.reshape(b, rshape);
        //     const ndrtb = NdArrayOf(rtb);
        //     const bb = broadCastedNdarray(ndrtb, backA.shape);
        //     nd_ops.add(backC, backA, bb);
        // }
        // c.name = `Tensor${c.id}_add_${a.id}_${b.id}`;
        // return c;
        return null;
    }

    neg(x: Tensor): Tensor {
        // const ndx = NdArrayOf(x);
        // const y = Tensor.create(null, x.shape, x.dtype);
        // const ndy = NdArrayOf(y);
        // nd_ops.neg(ndy, ndx);
        // y.name = `Tensor${y.id}_neg_${x.id}`;
        // return y;
        return null;
    }

    multiply(a: Tensor, b: Tensor): Tensor {
        // const nda = NdArrayOf(a);
        // const ndb = NdArrayOf(b);
        // const c = Tensor.create(null, a.shape, a.dtype);
        // const ndc = NdArrayOf(c);
        // nd_ops.multiply(ndc, nda, ndb);
        // c.name = `Tensor${c.id}_multiply_${a.id}_${b.id}`;
        // return c;
        return null;
    }

    relu(x: Tensor): Tensor {
        // const ndx = NdArrayOf(x);
        // const y = Tensor.create(null, x.shape, x.dtype);
        // const ndy = NdArrayOf(y);
        // nd_ops.maxs(ndy, ndx, 0);
        // y.name = `Tensor${y.id}_relu_${x.id}`;
        // return y;
        return null;
    }

    tile(x: Tensor, repeats: number[]): Tensor {
        return null;
    }

    pick(x: Tensor, indices: number[]): Tensor {
        return null;
    }

    // conv2d(
    //     x: Tensor,
    //     filter: Tensor,
    //     strides: number | [number, number],
    //     padding: number[],
    //     dataFormat: 'NHWC' | 'NCHW',
    //     dilations: number | [number, number] = 1,
    //     groups: number = 1): Tensor {

    //     if (!(Array.isArray(strides))) strides = [strides as number, strides as number];
    //     if (!(Array.isArray(dilations))) dilations = [dilations as number, dilations as number];

    //     let ndvx = NdArrayOf(x);  // 4d tensor, NHWC or NCHW
    //     let ndvk = NdArrayOf(filter); //[H, W, in, out] or [out, in, H, W]

    //     if (dataFormat == 'NCHW') {
    //         ndvx = ndvx.transpose([0, 2, 3, 1]);
    //         ndvk = ndvk.transpose([2, 3, 1, 0]);
    //     }

    //     ndvx = ndvx.pad([[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]]);
    //     ndvx = ndvx.rebuild();
    //     ndvk = ndvk.rebuild();

    //     const ndx = nd(ndvx.data, ndvx.shape);
    //     const ndk = nd(ndvk.data, ndvk.shape);

    //     // calc output shape
    //     const [batchSize, inputRows, inputCols, inputChannels] = [ndx.shape[0], ndx.shape[1], ndx.shape[2], ndx.shape[3]];
    //     const [kernelRows, kernelCols, kernelIn, outChannels] = [ndk.shape[0], ndk.shape[1], ndk.shape[2], ndk.shape[3]];
    //     ASSERT(inputChannels == kernelIn * groups, "intput channel do not match kernnels*groups");
    //     ASSERT(Math.floor(outChannels / groups) == outChannels / groups, "group can not equal devide outputChannels");

    //     const dilateKernelRows = kernelRows + (kernelRows - 1) * (dilations[0] - 1);
    //     const dilateKernelCols = kernelCols + (kernelCols - 1) * (dilations[1] - 1);
    //     const outRows = Math.floor((inputRows - dilateKernelRows + strides[0]) / strides[0]);
    //     const outCols = Math.floor((inputCols - dilateKernelCols + strides[1]) / strides[1]);
    //     const outSize = batchSize * outRows * outCols * outChannels;
    //     const resultTypedArray = new Float32Array(outSize);

    //     const sizeOfPatch = kernelRows * kernelCols * kernelIn;
    //     const patch = nd(new Float32Array(sizeOfPatch), [kernelRows, kernelCols, kernelIn]);

    //     const outChannelsPerGroup = outChannels / groups;
    //     const ndr = nd(resultTypedArray, [batchSize, outRows, outCols, outChannels]);
    //     const ndf = nd(new Float32Array(sizeOfPatch * outChannelsPerGroup), [sizeOfPatch, outChannelsPerGroup]);
    //     const pixelResult = nd(new Float32Array(outChannelsPerGroup), [1, outChannelsPerGroup]);

    //     for (let b = 0; b < batchSize; ++b) {
    //         let singleImage = ndx.pick(b, null, null, null);
    //         let singleResultOffset = b * (outRows * outCols * outChannels);

    //         for (let g = 0; g < groups; ++g) {
    //             for (let yC = g * outChannelsPerGroup, yCInGroup = 0; yCInGroup < outChannelsPerGroup; ++yCInGroup, ++yC) {
    //                 nd_ops.assign(patch, ndk.pick(null, null, null, yC));
    //                 const reshaped = nd(patch.data, [sizeOfPatch]);
    //                 nd_ops.assign(ndf.pick(null, yCInGroup), reshaped);
    //             }

    //             let offsetY = singleResultOffset + g * outChannelsPerGroup;
    //             for (let yH = 0; yH < outRows; ++yH) {
    //                 let xH = yH * strides[0];
    //                 for (let yW = 0; yW < outCols; ++yW) {
    //                     let xW = yW * strides[1];

    //                     let patchView = singleImage
    //                         .hi(xH + dilateKernelRows, xW + dilateKernelCols, (g + 1) * kernelIn)
    //                         .lo(xH, xW, g * kernelIn)
    //                         .step(dilations[0], dilations[1], 1);

    //                     nd_ops.assign(patch, patchView);
    //                     const patchInRow = nd(patch.data, [1, sizeOfPatch]);
    //                     nd_gemm(pixelResult, patchInRow, ndf);
    //                     ndr.data.set(pixelResult.data, offsetY);
    //                     offsetY += outChannels;
    //                 }
    //             }
    //         }
    //     }

    //     // [N, H, W, C] format
    //     let ndvr = new NdArray(resultTypedArray, [batchSize, outRows, outCols, outChannels]);
    //     if (dataFormat == "NCHW") {
    //         ndvr = ndvr.transpose([0, 3, 1, 2]);
    //     }
    //     const r = Tensor.fromBackend(new NdArrayTensor(ndvr, 'float32'));
    //     return r;
    // }

    nd_pad(x: nd, paddings: [number, number][], padVal: number = 0): nd {
        //TODO: check parameters
        const padTotal = paddings.reduce((sum, p) => sum + p[0] + p[1], 0);
        if (padTotal == 0) return x;
        const newShape = x.shape.map((len, index) => len + paddings[index][0] + paddings[index][1]);
        const newSize = newShape.reduce((mul, v) => mul * v, 1);
        const padded = nd(new Float32Array(newSize), newShape);
        padded.data.fill(0);
        const loPoint = paddings.map(beforeAndAfter => beforeAndAfter[0]);
        if (padVal != 0) {
            nd_ops.addseq(padded, padVal);
        }
        const sliced = padded.lo(...loPoint).hi(...x.shape);
        nd_ops.assign(sliced, x);
        return padded;
    }

    conv2d(
        x: Tensor,
        filter: Tensor,
        strides: number | [number, number],
        padding: number[],
        dataFormat: 'NHWC' | 'NCHW',
        dilations: number | [number, number] = 1,
        groups: number = 1): Tensor {

        if (!(strides instanceof Array)) strides = [strides as number, strides as number];
        if (!(dilations instanceof Array)) dilations = [dilations as number, dilations as number];
        let btx = backendTensorOf(x)._array.rebuild();
        let btk = backendTensorOf(filter)._array.rebuild();

        let ndx = nd(btx.data, btx.shape);  // 4d tensor, NHWC or NCHW
        let ndk = nd(btk.data, btk.shape); //[H, W, in, out] or [out, in, H, W]

        if (dataFormat == 'NCHW') {
            ndx = ndx.transpose(0, 2, 3, 1);
            ndk = ndk.transpose(2, 3, 1, 0);
        }
        ndx = this.nd_pad(ndx, [[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]]);
        // calc output shape
        const [batchSize, inputRows, inputCols, inputChannels] = [ndx.shape[0], ndx.shape[1], ndx.shape[2], ndx.shape[3]];
        const [kernelRows, kernelCols, kernelIn, outChannels] = [ndk.shape[0], ndk.shape[1], ndk.shape[2], ndk.shape[3]];
        ASSERT(inputChannels == kernelIn * groups, "intput channel do not match kernnels*groups");
        ASSERT(Math.floor(outChannels / groups) == outChannels / groups, "group can not equal devide outputChannels");

        const dilateKernelRows = kernelRows + (kernelRows - 1) * (dilations[0] - 1);
        const dilateKernelCols = kernelCols + (kernelCols - 1) * (dilations[1] - 1);
        const outRows = Math.floor((inputRows - dilateKernelRows + strides[0]) / strides[0]);
        const outCols = Math.floor((inputCols - dilateKernelCols + strides[1]) / strides[1]);
        const outSize = batchSize * outRows * outCols * outChannels;
        const resultTypedArray = new Float32Array(outSize);

        const sizeOfPatch = kernelRows * kernelCols * kernelIn;
        const patch = nd(new Float32Array(sizeOfPatch), [kernelRows, kernelCols, kernelIn]);

        const outChannelsPerGroup = outChannels / groups;
        const ndr = nd(resultTypedArray, [batchSize, outRows, outCols, outChannels]);
        const ndf = nd(new Float32Array(sizeOfPatch * outChannelsPerGroup), [sizeOfPatch, outChannelsPerGroup]);
        const pixelResult = nd(new Float32Array(outChannelsPerGroup), [1, outChannelsPerGroup]);

        for (let b = 0; b < batchSize; ++b) {
            let singleImage = ndx.pick(b, null, null, null);
            let singleResultOffset = b * (outRows * outCols * outChannels);

            for (let g = 0; g < groups; ++g) {
                for (let yC = g * outChannelsPerGroup, yCInGroup = 0; yCInGroup < outChannelsPerGroup; ++yCInGroup, ++yC) {
                    nd_ops.assign(patch, ndk.pick(null, null, null, yC));
                    const reshaped = nd(patch.data, [sizeOfPatch]);
                    nd_ops.assign(ndf.pick(null, yCInGroup), reshaped);
                }

                let offsetY = singleResultOffset + g * outChannelsPerGroup;
                for (let yH = 0; yH < outRows; ++yH) {
                    let xH = yH * strides[0];
                    for (let yW = 0; yW < outCols; ++yW) {
                        let xW = yW * strides[1];

                        let patchView = singleImage
                            .lo(xH, xW, g * kernelIn)
                            .hi(dilateKernelRows, dilateKernelCols, kernelIn)
                            .step(dilations[0], dilations[1], 1);

                        nd_ops.assign(patch, patchView);
                        const patchInRow = nd(patch.data, [1, sizeOfPatch]);
                        nd_gemm(pixelResult, patchInRow, ndf);
                        ndr.data.set(pixelResult.data, offsetY);
                        offsetY += outChannels;
                    }
                }
            }
        }

        // [N, H, W, C] format
        let ndvr = new NdArray(resultTypedArray, [batchSize, outRows, outCols, outChannels]);
        if (dataFormat == "NCHW") {
            ndvr = ndvr.transpose([0, 3, 1, 2]);
        }
        const r = Tensor.fromBackend(new NdArrayTensor(ndvr, 'float32'));
        return r;
    }
}

const backendName: string = "Backend_JSCPU";
const backendScore: number = 2;
const backendJsNdarray = new JsNdarrayBackend() as Backend;
ENV.registerBackend(backendName, backendJsNdarray, backendScore);
