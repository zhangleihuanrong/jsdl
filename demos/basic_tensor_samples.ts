import { tf } from '../src/index';
import { TensorPrintOptions } from '../src/backend';

const i = tf.tensor(null, [3, 10]);
tf.randomUniformEq(i, -1, 1);
tf.print(i);

const weight = tf.tensor(null, [10, 5]);
tf.randomNormEq(weight, 0, 10, 10000);
tf.print(weight, new TensorPrintOptions((x: number) => x.toExponential(5)));

const bias = tf.tensor(null, [5]);
tf.randomUniformEq(bias, -1, 1);
tf.print(tf.reshape(bias, [1, 5]));

const mul = tf.matMul(i
, weight);
tf.print(mul);
const linear = tf.add(mul, bias);
tf.print(linear);

const dense = tf.relu(linear);
tf.print(dense);

