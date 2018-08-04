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
  it("a[3x5] * b[5x2]", function() {
    const a = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5], [3, 5]);
    a.name = 'a_3x5';
    tf.print(a);

    const b = tf.tensor([1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5], [5, 2]);
    tf.print(b);
    
    const mul = tf.matMul(a, b, false, false);
    tf.print(mul);

    const gold = tf.tensor([15, 7.5, 40, 20, 15, 7.5], [3, 2]);
    gold.name = "Golden";
    tf.print(gold);

    assert(areTwoArrayLikeEqual(mul.shape, gold.shape), "result shape error!");
    assert(areTwoArrayLikeEqual(tf.readSync(gold), tf.readSync(mul)), "result data error!");
  });

  it("a[5x3].transpose.transpose.reshape([3x5]) * b[5x2]", function() {
    let a = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5], [5, 3]);
    a.name = 'a_5x3';
    tf.print(a);

    a = tf.transpose(a, [1, 0]);
    a.name = 'aTranspose_3x5';
    tf.print(a);

    a = tf.transpose(a, [1, 0]);
    a.name = 'aTT_5x3';
    tf.print(a);

    a = tf.reshape(a, [3, 5]);
    a.name = 'aTTReshape_3x5';
    tf.print(a);

    const b = tf.tensor([1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5], [5, 2]);
    tf.print(b);
    
    const mul = tf.matMul(a, b, false, false);
    tf.print(mul);

    const gold = tf.tensor([15, 7.5, 40, 20, 15, 7.5], [3, 2]);
    gold.name = "Golden";
    tf.print(gold);

    assert(areTwoArrayLikeEqual(mul.shape, gold.shape), "result shape error!");
    assert(areTwoArrayLikeEqual(tf.readSync(gold), tf.readSync(mul)), "result data error!");
  });

  it("a[1x3x5].tile(2,1,1) * b[5x2]", function() {
    let a = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5], [1, 3, 5]);
    a = tf.tile(a, [2, 1, 1]);
    a.name = 'a_Tile_2x3x5';
    tf.print(a);

    let b = tf.tensor([1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5], [1, 5, 2]);
    b = tf.transpose(b, [2, 1, 0]);
    tf.print(b);
    
    let mul = tf.matMul(a, b, false, true);
    mul.name = 'matMul result';
    tf.print(mul);

    let gold = tf.tensor([15, 7.5, 40, 20, 15, 7.5], [1, 3, 2]);
    gold = tf.tile(gold, [2, 1, 1]);
    gold.name = "Golden";
    tf.print(gold);

    assert(areTwoArrayLikeEqual(mul.shape, gold.shape), "result shape error!");
    assert(areTwoArrayLikeEqual(tf.readSync(gold), tf.readSync(mul)), "result data error!");
  });

  it("a[1024*1024] * b[1024x1024]", function() {
    const startInitA = new Date().getTime();
    let a = tf.randomNorm([2048, 2048], 2.0, 3.0, 'float32', 10000);
    const msInitA = (new Date()).getTime() - startInitA;
    console.log(`  Finish initialize A 2048 in ${msInitA}ms.`);
    //tf.print(a, null, [3, 2], [3, 2]);

    const startInitB = new Date().getTime();
    let b = tf.randomNorm([2048, 2048], 1.0, 2.0, 'float32', 20000);
    let msInitB = (new Date()).getTime() - startInitB;
    console.log(`  Finish initialize B 2048 in ${msInitB}ms.`);
    //tf.print(b, null, [3, 2], [3, 2]);
    
    const startMatMul = new Date().getTime();
    console.log('start mat mul............');
    let mul = tf.matMul(a, b, false, false);
    let msMatMul = (new Date()).getTime() - startMatMul;
    console.log(`  Finish MatMul in ${msMatMul}ms. result is of shape: ${mul.shape}`);

    assert(true, "");
  }).timeout(200000);

});
