import { ENV } from './environments';
import { Tensor } from './tensor';
import { print } from './utils';

const transpose =  ENV.engine.backend.transpose;
const matMul = ENV.engine.backend.matMul;
const neg = ENV.engine.backend.neg;
const add = ENV.engine.backend.add;
const multiply = ENV.engine.backend.multiply;
const relu = ENV.engine.backend.relu;
const readSync = ENV.engine.backend.readSync;
const reshape = ENV.engine.backend.reshape;
const randomNormEq = ENV.engine.backend.randomNormEq;
const randomUniformEq = ENV.engine.backend.randomUniformEq;
const conv2d = ENV.engine.backend.conv2d;


const tensor = Tensor.create;

export { tensor, print, transpose, conv2d, matMul, neg, add, multiply, relu, readSync, reshape, randomNormEq, randomUniformEq};
