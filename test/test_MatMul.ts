import { assert } from 'chai';

import * as tf from '../src/index';

function areTwoArrayLikeEqual<T extends number[] | Float32Array | Int32Array | Uint8Array>(a: T, b:T) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

describe("Tensor MatMul", function() {
  it("a[2x5] * b[5x2]", function() {
    let a = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5], [5, 3]);
    a.name = 'a_5x3';
    tf.print(a);

    a = tf.transpose(a, [1,0]);
    a.name = 'aTranspose_3x5';
    tf.print(a);
    
    const b = tf.tensor([1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5], [5, 2]);
    tf.print(b);
    
    const mul = tf.matMul(a, b);
    tf.print(mul);

    const gold = tf.tensor([15, 7.5, 40, 20, 15, 7.5], [3, 2]);
    gold.name = "Golden";
    tf.print(gold);

    assert(areTwoArrayLikeEqual(mul.shape, gold.shape), "result shape error!");
    assert(areTwoArrayLikeEqual(tf.readSync(gold), tf.readSync(mul)), "result data error!");
  });
});
