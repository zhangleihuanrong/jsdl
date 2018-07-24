import { assert } from 'chai';

import * as tf from '../src/index';

describe("Basic Tensor", function() {
  it("Should work", function() {
    const i = tf.randomUniform([3, 10], -1, 1);
    tf.print(i);
    
    const weight = tf.randomNorm([10, 5], 0, 10, 'float32', 30000);
    tf.print(weight, (x: number) => x.toExponential(5));
    
    const bias = tf.tensor([1, 2, 3, 4, 5]);
    tf.reshape(bias, [1, 5]).print();
    
    const mul = tf.matMul(i, weight);
    mul.print(); // or tf.print(mul);
    
    const linear = tf.add(mul, bias);
    tf.print(linear);
    
    const dense = tf.relu(linear);
    tf.print(dense);

    assert(true == true);
  });
});
