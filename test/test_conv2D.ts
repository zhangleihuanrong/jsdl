import axios from 'axios';

async function loadTensor(url: string, shape: number[]) : Promise<Tensor> {
    try {
        const response = await axios.request({
            responseType: 'arraybuffer',
            url: url,
            method: 'get',
            headers: { 'Content-Type': 'application/octet-stream' },
        });
        const buffer = response.data as ArrayBuffer;
        const flen = buffer.byteLength / 4;
        const ta = new Float32Array(flen);
        const dv = new DataView(buffer);
        for (let i = 0; i < flen; ++i) {
            ta[i] = dv.getFloat32(i * 4, true);
        }
        return tf.tensor(ta, shape);
    }
    catch(e) {
        throw e;
    }
}


import { assert } from 'chai';

import * as tf from '../src/index';
import {Tensor} from '../src/index';

import { NDView as NdArray} from '../src/NdView/ndview';

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
        
    goldenResult.name = `GoldConv2DResult${goldenResult.id}`;
    tf.print(goldenResult, number2string, lastAxisExclude, otherAxisExclude);
    console.log(`padding:${padding}, strides:${strides}, dilations:${dilations}`);
    let r: Tensor = null;
    const rounds = 3;
    const start =  Date.now();
    for (let rep=0; rep < rounds; ++rep) {
        const itStart =  Date.now();
        r = tf.conv2d(image, filter, strides, padding, 'NCHW', dilations);
        const millis = Date.now() - itStart;
        console.log(`  --${rep} iteration: ${millis}ms`);
    }
    
    const millis = Date.now() - start;
    const avgMillis = millis / rounds;
    console.log(`  --Total: ${millis}ms, avg:${avgMillis}ms`);
    
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
    it("x[1x1x4x4, k[2x1x2x2], nopadding, strides=[2,2]", function() {
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
        testConv2D(
            tf.tensor(ima, [1, 1, 4, 4]),
            tf.tensor(flta, [2, 1, 2, 2]),
            tf.tensor(goldb, [1, 2, 3, 3]),
            [1, 1, 1, 1],
            [2, 2],
            [1, 1]
        );
    });


    it("x[1x3x224x224, k[64x3x7x7], pad[3,3,3,3], strides=[2,2]", async function() {
        const pX = loadTensor("./testdata/imageInput.buf", [1, 3, 224, 224]);
        const pF = loadTensor("./testdata/filter.buf", [64, 3, 7, 7]);
        const pGold = loadTensor("./testdata/convResult.buf", [1, 64, 112, 112]);

        Promise.all([pX, pF, pGold])
            .then((tensors: Tensor[]) => {
                testConv2D(
                    tensors[0], tensors[1], tensors[2],
                    [3, 3, 3, 3],
                    [2, 2],
                    [1, 1],
                    (x: number) => x.toFixed(7),
                    [5, -3],
                    [2, -1]
                );
            });
    });
});

