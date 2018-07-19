import * as fs from 'fs';

import * as tf from '../src/index';
import {Tensor} from '../src/index';

import { iterateNdarray } from '../src/utils/ndarray_print';

import * as ndarray from 'ndarray';

function loadTensor(pathName: string, shape: number[]) : Tensor {
    const buffer = fs.readFileSync(pathName);
    const flen = buffer.byteLength / 4;
    const ta = new Float32Array(flen);
    const dv = new DataView(buffer.buffer);
    for (let i = 0; i < flen; ++i) {
        ta[i] = dv.getFloat32(i * 4, true);
    }

    return tf.tensor(ta, shape);
}

const number2string = (x: number) => x.toExponential(5);
const lastAxisExclude=  [3, -2] as [number, number];

function isNumberNotSame(a: number, b: number) : Boolean {
    const aa = Math.abs(a);
    const ab = Math.abs(b);
    return ((Math.max(aa, ab) >= 1e-6) && (Math.abs(a - b)/Math.max(aa, ab) > 0.15));
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
    dilations: [number, number]
) {
    image.name = `image${image.id}`;
    tf.print(image, number2string, lastAxisExclude);

    filter.name = `filter${filter.id}`;
    tf.print(filter, number2string, lastAxisExclude);
        
    goldenResult.name = `GoldConv2DResult${goldenResult.id}`;
    tf.print(goldenResult, number2string, lastAxisExclude);
        
    let r: Tensor = null;
    const rounds = 1;
    const start =  Date.now();
    for (let rep=0; rep < rounds; ++rep) {
        const itStart =  Date.now();
        r = tf.conv2d(image, filter, strides, padding, 'NCHW', dilations);
        const millis = Date.now() - itStart;
        console.log(`  --${rep} iteration: ${millis}ms`);
    }
    
    const millis = Date.now() - start;
    const avgMillis = millis / rounds;
    
    r = tf.transpose(r, [0, 3, 1, 2]);
    r.name = `Conv2dResult${r.id}`;
    tf.print(r, number2string, lastAxisExclude);
 
    console.log(`  --Total: ${millis}ms, avg:${avgMillis}ms`);
 
    if (!arraysEqual(goldenResult.shape, r.shape)) {
        throw new Error('Not same shape, gold:' + JSON.stringify(goldenResult.shape) + " .vs. target:" + JSON.stringify(r.shape));
    }
    
    // TODO: add tensor iteration interface
    // currently hacked with ndarray
    const ga: ndarray = goldenResult.data as ndarray;
    const ra: ndarray = r.data as ndarray;
    
    iterateNdarray(ga, (nda: ndarray, loc) => {
        const gv = ga.get(...loc);
        const rv = ra.get(...loc);
        if (isNumberNotSame(gv, rv)) {
            throw new Error(`not equal @`+ JSON.stringify(loc) + `  gold:${gv}, r:${rv}`);
        }
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
  
testConv2D(
    tf.tensor(ima, [1, 1, 4, 4]),
    tf.tensor(flta, [2, 1, 2, 2]),
    tf.tensor(golda, [1, 2, 2, 2]),
    [0, 0, 0, 0],
    [2, 2],
    [1, 1]
);

testConv2D(
    tf.tensor(ima, [1, 1, 4, 4]),
    tf.tensor(flta, [2, 1, 2, 2]),
    tf.tensor(goldb, [1, 2, 3, 3]),
    [1, 1, 1, 1],
    [2, 2],
    [1, 1]
);

testConv2D(
    loadTensor("testdata/imageInput.buf", [1, 3, 224, 224]),
    loadTensor("testdata/filter.buf", [64, 3, 7, 7]),
    loadTensor("testdata/convResult.buf", [1, 64, 112, 112]),
    [3, 3, 3, 3],
    [2, 2],
    [1, 1]
);
