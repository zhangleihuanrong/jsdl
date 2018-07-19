import * as tf from '../src/index';

const i = tf.randomUniform([3, 10], -1, 1);
tf.print(i);

const weight = tf.randomNorm([10, 5], 0, 10, 'float32', 30000);
tf.print(weight, (x: number) => x.toExponential(5));

const bias = tf.randomUniform([5], -1, 1);
tf.reshape(bias, [1, 5]).print();

const mul = tf.matMul(i, weight);
mul.print(); // or tf.print(mul);

const linear = tf.add(mul, bias);
tf.print(linear);

const dense = tf.relu(linear);
tf.print(dense);

