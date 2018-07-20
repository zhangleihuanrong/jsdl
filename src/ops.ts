import { ENV } from './environments';
import { Tensor } from './tensor';

function printTensor(
    x: Tensor,
    number2string: (x: number) => string = null, 
    excludeLastAxis: [number, number] = null,
    excludeHiAxises: [number, number] = null) 
{
    x.print(number2string, excludeLastAxis, excludeHiAxises);
}

const transpose =  ENV.engine.backend.transpose;
const matMul = ENV.engine.backend.matMul;
const neg = ENV.engine.backend.neg;
const add = ENV.engine.backend.add;
const multiply = ENV.engine.backend.multiply;
const relu = ENV.engine.backend.relu;
const readSync = ENV.engine.backend.readSync;
const reshape = ENV.engine.backend.reshape;
const conv2d = ENV.engine.backend.conv2d;

// static Tensor methods
const tensor = Tensor.create;
const randomNorm = Tensor.randomNorm;
const randomUniform = Tensor.randomUniform;
const truncatedNorm = Tensor.truncatedNorm;

export { tensor, randomNorm, randomUniform, truncatedNorm,
    
        readSync,

        printTensor as print, 
        
        transpose, reshape, 
        
        matMul, neg, add, multiply, relu,

        conv2d };
