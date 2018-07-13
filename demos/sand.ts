
// import * as path from 'path';
// console.log(path);

import * as ndarray from 'ndarray';

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
ndarray_print2(a);
for (let n = 0; n < 3; ++n) {
    const col = a.pick([null, n]);
    console.log('colum', n, col.shape );
}

const b = a.transpose(1, 0);
ndarray_print2(b);

a.set(0, 0, 10000);
ndarray_print2(a);
ndarray_print2(b);

const fa = new Float32Array(50);
const shape = [ 5, 10];
const c = ndarray(fa, shape);
ndarray_print2(c);

import '../src/backends/backend_js';
import { tf } from '../src/index';

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

const image = tf.tensor(null, [1, 3, 242, 242]);
tf.randomUniformEq(image, -1.0, 1.0);

const filter = tf.tensor(null, [128, 3, 7, 7]);
tf.randomNormEq(filter, 0, 1, 10000);

const padding = [1, 1, 1, 1];

const r = tf.conv2d(image, filter, 1, padding, 'NCHW', 1);
console.log("==== r.shape ", r.shape);
