import { assert } from 'chai';

import * as tf from '../src';

describe("Operators", function() {
    it("BatchNormalize   (float)", function() {
        const X = tf.tensor([1.0, 2.0, -1.0, -2.0, -3.0, 0.0], [1, 2, 3]).setName('X');
        tf.print(X);

        const S = tf.tensor([1.0, 2.0], [2]).setName('Scale');
        tf.print(S);

        const B = tf.tensor([10.0, 20.0], [2]).setName('Bias');
        tf.print(B);

        const M = tf.tensor([-1.0, 1.5], [2]).setName('Mean');
        tf.print(M);

        const V = tf.tensor([4.0, 9.0], [2]).setName('Variance');
        tf.print(V);

        const bn = tf.batchNormalize(X, S, B, M, V, 1e-8, 0, 0).setName('BatchNormalized');
        tf.print(bn);

        assert(true);
    });

    it("Reshape (float)", function() {
        const X1 = tf.tensor([1.0, 2.0, -1.0, -2.0, -3.0, 0.0], [2, 3]);
        tf.print(X1.setName('X1'));

        const transX1 = tf.transpose(X1, [1, 0]);
        tf.print(transX1.setName('transX1'));

        const reshapeX1 = tf.reshape(transX1, [1, -1, 6]);
        tf.print(reshapeX1.setName('reshapeX1'));
    });
});

