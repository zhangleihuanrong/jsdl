import axios from 'axios';
import { assert } from 'chai';

import { ENV } from '../src/environments';
ENV.preferBackend("Backend_JSCPU");

import { NDView as NdArray} from '../src/NdView/ndview';
import * as tf from '../src';
import {Tensor} from '../src';

export function loadTensor(url: string, shape: number[]) : Promise<Tensor> {
    return new Promise((resolve, reject) => {
        axios.request({
            responseType: 'arraybuffer',
            url: url,
            method: 'get',
            headers: { 'Content-Type': 'application/octet-stream' },
        }).then(response => {
            const buffer = response.data as ArrayBuffer;
            const flen = buffer.byteLength / 4;
            const ta = new Float32Array(flen);
            const dv = new DataView(buffer);
            for (let i = 0; i < flen; ++i) {
                ta[i] = dv.getFloat32(i * 4, true);
            }
            resolve(tf.tensor(ta, shape));
        }).catch(err => {
            reject(err);
        });
    });
}

function isNumberNotSame(a: number, b: number) : Boolean {
    const aa = Math.abs(a);
    const ab = Math.abs(b);
    return ((Math.abs(a - b) >= 1e-4) && (Math.abs(a - b)/(Math.max(aa, ab)+Math.abs(a - b)) > 0.01));
}

function arraysEqual(a: number[], b: number[]) : boolean {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;

    for (let i = 0; i < a.length; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}


function testConv2D(
    image: Tensor, 
    filter: Tensor, 
    goldenResult: Tensor, 
    padding:number[], 
    strides: [number, number], 
    dilations: [number, number],
    number2string: (number) => string = (x) => x.toString(),
    lastAxisExclude: [number, number] = null,
    otherAxisExclude: [number, number] = null
) {
    image.name = `image${image.id}`;
    tf.print(image, number2string, lastAxisExclude, otherAxisExclude);

    filter.name = `filter${filter.id}`;
    tf.print(filter, number2string, lastAxisExclude, otherAxisExclude);
    console.log(`----------padding:${padding}, strides:${strides}, dilations:${dilations}`);
        
    let r: Tensor = null;
    const rounds = 6;
    const start =  Date.now();
    for (let rep=0; rep < rounds; ++rep) {
        const itStart =  Date.now();
        r = tf.conv2d(image, filter, strides, padding, 'NCHW', dilations);
        //tf.read(r);
        const millis = Date.now() - itStart;
        console.log(`  --${rep} iteration: ${millis}ms, result elements: ${r.shape}`);
    }
    const millis = Date.now() - start;
    const avgMillis = millis / rounds;
    console.log(`  --Total: ${millis}ms in ${rounds} iterations, avg:${avgMillis}ms`);

    goldenResult.name = `GoldConv2DResult${goldenResult.id}`;
    tf.print(goldenResult, number2string, lastAxisExclude, otherAxisExclude);

    r.name = `Conv2dResult${r.id}`;
    tf.print(r, number2string, lastAxisExclude, otherAxisExclude);
 
    if (!arraysEqual(goldenResult.shape, r.shape)) {
        throw new Error('Not same shape, gold:' + JSON.stringify(goldenResult.shape) + " .vs. target:" + JSON.stringify(r.shape));
    }
    
    const ga: NdArray = new NdArray(tf.read(goldenResult), goldenResult.shape);
    const ra: NdArray = new NdArray(tf.read(r), r.shape);
    
    ga.forEach((gv, index, loc) => {
        const rv = ra.get(...loc);
        assert(!isNumberNotSame(gv, rv), `not equal @`+ JSON.stringify(loc) + `  gold:${gv}, r:${rv}`);
     });
}


const ima = [
    -0.5, +0.5, +1.0, +1.0, 
    +0.5, +1.0, +0.5, +1.0,
    +1.0, +1.0, -0.5, -0.5,
    +1.0, -1.0, -1.0, +0.0
];

const flta = [
    1, 0,
    0, 1,

    0, 1,
    1, 0
];

const golda = [
  0.5,  2.0, 
  0.0, -0.5,

  1.0,  1.5,
  2.0, -1.5
];

const goldb = [ // for padding [1,1,1,1]
    -0.5, +1.0, +0.0, 
    +1.0, +0.5, +1.0,
    +0.0, -1.0, +0.0,
  
    +0.0, +0.5, +1.0,
    +0.5, +1.5, -0.5,
    +1.0, -1.0, +0.0
];

describe("Conv2D", function() {
    it("SUM (float)", function() {
        const X1 = tf.tensor([1.0, 2.0, -1.0, -2.0, -3.0, 0.0], [2, 3]);
        tf.print(X1.setName('X1'));

        const logSumExpX1 = tf.logSumExp(X1, [1]);
        tf.print(logSumExpX1.setName('logSumExpX1'));

        const x_logSumExp = tf.sub(X1, logSumExpX1);
        tf.print(x_logSumExp.setName('x-logSumExp'));

        const xSoftMax = tf.exp(x_logSumExp);
        tf.print(xSoftMax.setName('xSoftMax'));

        const softmaxX1Sum = tf.sum(xSoftMax);
        tf.print(softmaxX1Sum.setName('softmaxX1Sum'));

        let ta = tf.read(softmaxX1Sum);
        assert(Math.abs(ta[0] - 1.0) < 0.001 && Math.abs(ta[1] - 1.0) < 0.001);

        const dirSoftmaxX1 = tf.softmax(X1);
        tf.print(dirSoftmaxX1.setName('directSoftMax'));

        const dirSoftmaxX1Sum = tf.sum(xSoftMax);
        tf.print(dirSoftmaxX1Sum.setName('dirSoftmaxX1Sum'));

        ta = tf.read(dirSoftmaxX1Sum);
        assert(Math.abs(ta[0] - 1.0) < 0.001 && Math.abs(ta[1] - 1.0) < 0.001);

    }).timeout(10000);

    it("RELU,EXP", function() {
        const X1 = tf.tensor([1.0, 2.0, -1.0, -2.0, -3.0, 0.0], [2, 3]);
        tf.print(X1);
        const ReluX1 = tf.relu(X1);
        tf.print(ReluX1);
        const expX1 = tf.exp(X1);
        tf.print(expX1);
    }).timeout(10000);

    it("x[1x3x224x224, k[64x3x7x7], pad[3,3,3,3], strides=[2,2]", async function() {
        console.log(`===========Using backend: ${ENV.getCurrentBackendName()} ============`);
        console.log('Downloading imageInput.buf...');
        const pX = await loadTensor("./testdata/imageInput.buf", [1, 3, 224, 224]);
        console.log('Downloading filter.buf...');
        const pF = await loadTensor("./testdata/filter.buf", [64, 3, 7, 7]);
        console.log('Downloading goldresult...');
        const pGold = await loadTensor("./testdata/convResult.buf", [1, 64, 112, 112]);
        console.log('start n-time conv2d...');

        testConv2D(
            pX, pF, pGold,
            [3, 3, 3, 3],
            [2, 2],
            [1, 1],
            (x: number) => x.toFixed(7),
            [5, -3],
            [2, -1]
        );
    }).timeout(100000);

    it("x[1x1x4x4, k[2x1x2x2], nopadding, strides=[2,2]", function() {
        console.log(`===========Using backend: ${ENV.getCurrentBackendName()} ============`);
        testConv2D(
            tf.tensor(ima, [1, 1, 4, 4]),
            tf.tensor(flta, [2, 1, 2, 2]),
            tf.tensor(golda, [1, 2, 2, 2]),
            [0, 0, 0, 0],
            [2, 2],
            [1, 1]
        );
    });

    it("x[1x1x4x4, k[2x1x2x2], pad[1,1,1,1], strides=[2,2]", function() {
        console.log(`===========Using backend: ${ENV.getCurrentBackendName()} ============`);
        testConv2D(
            tf.tensor(ima, [1, 1, 4, 4]),
            tf.tensor(flta, [2, 1, 2, 2]),
            tf.tensor(goldb, [1, 2, 3, 3]),
            [1, 1, 1, 1],
            [2, 2],
            [1, 1]
        );
    });
});

