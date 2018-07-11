
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
const a = ndarray([1, 2, 3, 2, 2, 3], [2, 3]);
ndarray_print2(a);

const b = a.transpose(1, 0);
ndarray_print2(b);

a.set(0, 0, 10000);
ndarray_print2(a);
ndarray_print2(b);

const fa = new Float32Array(50);
const shape = [ 5, 10];
const c = ndarray(fa, shape);
ndarray_print2(c);

import { tf } from '../src/index';

const i = tf.tensor(null, [3, 100]);
const weight = tf.tensor(null, [100, 20]);
const bias = tf.tensor(null, [20]);
const linear = tf.add(tf.matMul(i, weight), bias);
const dense = tf.relu(linear);

tf.print(dense);
