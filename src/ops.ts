import { Tensor } from './tensor';
import { ENV } from './environments';
// import { Tensor } from './tensor';
// import { Backend } from './backend';
// import { TensorManager } from './tensor_manager';

const transpose =  ENV.engine.backend.transpose;
const matMul = ENV.engine.backend.matMul;
const neg = ENV.engine.backend.neg;
const add = ENV.engine.backend.add;
const multiply =  ENV.engine.backend.multiply;
const relu = ENV.engine.backend.relu;
const tensor = Tensor.create;

export { tensor, transpose, matMul, neg, add, multiply, relu };
