import { assert } from 'chai';

import * as tf from '../src';
import { areArraysEqual, areArraysNearEnough } from '../src/utils/gadget';

describe("Gemm Operators", function() {
    it("Gemm pool with 2x2, strides 2x2", function() {
        const a = tf.reshape(tf.range(0, 15, 1.0, 'float32'), [3, 5]).setName('a');
        tf.print(a);
        const b = tf.tensor(new Array(10).fill(1.0), [5, 2]).setName('b');
        tf.print(b);
        const bias = tf.tensor([0.5, 0.75], [2]).setName('bias');
        tf.print(bias);
        const gemm = tf.gemm(a, b, bias).setName('GemmResult');
        tf.print(gemm);
        assert(areArraysEqual(gemm.shape, [3, 2]), "result shape error!");
        assert(areArraysNearEnough(tf.read(gemm), [10.5, 10.75, 35.5, 35.75, 60.5, 60.75]), "result value error!");
    });
});

