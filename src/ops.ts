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

const transpose =  ENV.engine.backend.transpose.bind(ENV.engine.backend);
const matMul = ENV.engine.backend.matMul.bind(ENV.engine.backend);
const neg = ENV.engine.backend.neg.bind(ENV.engine.backend);
const add = ENV.engine.backend.add.bind(ENV.engine.backend);
const multiply = ENV.engine.backend.multiply.bind(ENV.engine.backend);
const relu = ENV.engine.backend.relu.bind(ENV.engine.backend);
const readSync = ENV.engine.backend.readSync.bind(ENV.engine.backend);
const reshape = ENV.engine.backend.reshape.bind(ENV.engine.backend);
const conv2d = ENV.engine.backend.conv2d.bind(ENV.engine.backend);

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
