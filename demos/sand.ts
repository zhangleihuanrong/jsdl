
// import * as path from 'path';
// console.log(path);

import * as ndarray from 'ndarray';
import * as nd_ops from 'ndarray-ops';
import * as fs from 'fs';

function ndarray_print2(arr) {
    console.log(`  ==shape:${arr.shape}`)
    for (let y = 0; y < arr.shape[0]; ++y) {
        let line = "";
        for (let x = 0; x < arr.shape[1]; ++x) {
            line += " " + arr.get(y, x) ;
        }
        console.log("...   ", line);
    }
}
const a = ndarray(new Float32Array([1, 2, 3, 2, 2, 3]), [2, 3]);
a.set(0, 0, 10000);
a.set(1, 2, 20000);
ndarray_print2(a);

const b = ndarray(new Float32Array(6), [2, 3]);
ndarray_print2(b);
nd_ops.assign(b, a);
ndarray_print2(b);

const fa = new Float32Array(50);
const shape = [ 5, 10 ];
const c = ndarray(fa, shape);
ndarray_print2(c);

import '../src/backends/backend_js';
import { tf, Tensor } from '../src/index';

function loadTensor(pathName: string, shape: number[]) : Tensor {
    const buffer = fs.readFileSync(pathName);
    const ta = new Float32Array(buffer);
    return tf.tensor(ta, shape);
}

const i = tf.tensor(null, [3, 10]);
tf.randomUniformEq(i, -1, 1);
tf.print(i, 'i');

const weight = tf.tensor(null, [10, 5]);
tf.randomNormEq(weight, 0, 10, 10000);
tf.print(weight, 'weight');

const bias = tf.tensor(null, [5]);
tf.randomUniformEq(bias, -1, 1);
tf.print(tf.reshape(bias, [1, 5]), 'bias');

const mul = tf.matMul(i, weight);
tf.print(mul, 'mul');

const linear = tf.add(mul, bias);
tf.print(linear, 'linear');

const dense = tf.relu(linear);
tf.print(dense, 'dense');

const image = loadTensor("testdata/imageInput.buf", [1, 3, 224, 224]);
const filter = loadTensor("testdata/filter.buf", [64, 3, 7, 7]);
const goldenResult = loadTensor("test/data/convResult.buf", [1, 64, 112, 112]);

const padding = [3, 3, 3, 3];
const strides : [number, number] = [2, 2];
const dilations: [number, number] = [1, 1];

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

console.log(r.shape);

function arraysEqual(a: number[], b: number[]) : bool {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;

    for (let i = 0; i < a.length; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

if (!arraysEqual(goldenResult.shape, r.shape)) throw new Error('Not equal');
const ga = tf.readSync(goldenResult);
const ra = tf.readSync(r);

let i = 0;
for (let b = 0; b < goldenResult.shape[0]; ++b) {
    for (let c = 0; c < goldenResult.shape[1]; ++c) {
        for (let h = 0; h < goldenResult.shape[2]; ++h) {
            for (let w = 0; w < goldenResult.shape[3]; ++w) {
                const gv = ga[i];
                const rv = ra[i];
                if (gv - rv > 0.01) {
                    throw new Error('not equal');
                }
                i++;
            }
        }
    }
}

