import { assert } from 'chai';

import * as tf from '../src';
import { areArraysEqual } from '../src/utils/gadget';

describe("Pooling Operators", function() {
    it("Max Pooling 4x4 pool with 2x2, strides 2x2", function() {
        const X = tf.reshape(tf.range(0, 16, 1.0, 'float32'), [1, 1, 4, 4]).setName('X');
        tf.print(X);
        const mp = tf.maxPool(X, [2, 2], [2, 2], [0, 0, 0, 0], 1).setName('MaxPoolResult');
        tf.print(mp);
        assert(areArraysEqual(mp.shape, [1, 1, 2, 2]), "result shape error!");
        assert(areArraysEqual(tf.read(mp), [5, 7, 13, 15]), "result value error!");
    });

    it("Max Pooling 4x4 pool with 2x2, strides 2x2, paddding 1", function() {
        const X = tf.reshape(tf.range(1, 17, 1.0, 'float32'), [1, 1, 4, 4]).setName('X');
        tf.print(X);
        const mp = tf.maxPool(X, [2, 2], [2, 2], [1, 1, 1, 1], 1).setName('MaxPoolResult');
        tf.print(mp);
        assert(areArraysEqual(mp.shape, [1, 1, 3, 3]), "result shape error!");
        assert(areArraysEqual(tf.read(mp), [1, 3, 4, 9, 11, 12, 13, 15, 16]), "result value error!");
    });

    it("Average Pooling 4x4 pool with 2x2, strides 2x2", function() {
        const X = tf.reshape(tf.range(0, 16, 1.0, 'float32'), [1, 1, 4, 4]).setName('X');
        tf.print(X);

        const mp = tf.averagePool(X, [2, 2], [2, 2], [0, 0, 0, 0], 1).setName('AvgPoolResult');
        tf.print(mp);

        assert(areArraysEqual(mp.shape, [1, 1, 2, 2]), "result shape error!");
        assert(areArraysEqual(tf.read(mp), [2.5, 4.5, 10.5, 12.5]), "result value error!");
    });

    it("Average Pooling 4x4 pool with 2x2, strides 2x2, padding 1", function() {
        const X = tf.reshape(tf.range(1, 17, 1.0, 'float32'), [1, 1, 4, 4]).setName('X');
        tf.print(X);

        const mp = tf.averagePool(X, [2, 2], [2, 2], [1, 1, 1 , 1], 1).setName('AvgPoolResult');
        tf.print(mp);

        assert(areArraysEqual(mp.shape, [1, 1, 3, 3]), "result shape error!");
        assert(areArraysEqual(tf.read(mp), [0.25, 1.25, 1, 3.5, 8.5, 5, 3.25, 7.25, 4]), "result value error!");
    });

    it("Average Pooling 4x4 pool with 2x2, strides 2x2, padding 1, count not include pad", function() {
        const X = tf.reshape(tf.range(1, 17, 1.0, 'float32'), [1, 1, 4, 4]).setName('X');
        tf.print(X);

        const mp = tf.averagePool(X, [2, 2], [2, 2], [1, 1, 1 , 1], 0).setName('AvgPoolResult');
        tf.print(mp);

        assert(areArraysEqual(mp.shape, [1, 1, 3, 3]), "result shape error!");
        assert(areArraysEqual(tf.read(mp), [1, 2.5, 4, 7, 8.5, 10, 13, 14.5, 16]), "result value error!");
    });

});

