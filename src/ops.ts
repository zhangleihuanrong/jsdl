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
const reshape = ENV.engine.backend.reshape.bind(ENV.engine.backend);
const conv2d = ENV.engine.backend.conv2d.bind(ENV.engine.backend);
const tile =  ENV.engine.backend.tile.bind(ENV.engine.backend);
const pick =  ENV.engine.backend.pick.bind(ENV.engine.backend);
const read = ENV.engine.backend.read.bind(ENV.engine.backend);
const batchNormalize = ENV.engine.backend.batchNormalize.bind(ENV.engine.backend);
const maxPool = ENV.engine.backend.maxPool.bind(ENV.engine.backend);
const averagePool = ENV.engine.backend.averagePool.bind(ENV.engine.backend);
const gemm = ENV.engine.backend.gemm.bind(ENV.engine.backend);
const softmax = ENV.engine.backend.softmax.bind(ENV.engine.backend);


// static Tensor methods
const tensor = Tensor.create;
const randomNorm = Tensor.randomNorm;
const randomUniform = Tensor.randomUniform;
const truncatedNorm = Tensor.truncatedNorm;

export { tensor, randomNorm, randomUniform, truncatedNorm,
    
        read, 

        printTensor as print, 
        
        transpose, reshape, tile, pick,
        
        relu, neg,

        matMul, add, multiply,

        conv2d, batchNormalize, maxPool, averagePool, gemm, softmax };
