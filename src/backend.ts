import { TypedArray } from './types';
import { Tensor} from './tensor';
import { TensorManager } from './tensor_manager';


export interface Backend extends TensorManager{
    //write(t: Tensor, values: StrictTensorLike) : void;
    //read(t: Tensor) : Promise<TypedArray>;
    read(t: Tensor) : TypedArray;

    conv2d(
        x: Tensor, 
        filter: Tensor,
        strides: number | [number, number], 
        padding: number[], 
        dataFormat: 'NHWC' | 'NCHW', 
        dialations: number | [number, number],
        groups: number,
        bias: Tensor) : Tensor;
    
    // x: [N, C, H, W, ......]
    // scale: [C]
    // bias: [C]
    // mean: [C], running mean in training
    // variance: [C], running variance in training
    batchNormalize(x: Tensor, scale: Tensor, bias: Tensor, mean: Tensor, variance: Tensor,
        epsilon: number, momentum : number, spatial: number) : Tensor;

    // x: [N, C, H, W,......]
    maxPool(x: Tensor, kernelShape: number[], strides: number[], pads: number[], storageOrder: number) : Tensor;

    // x: [N, C, H, W,......]
    averagePool(x: Tensor, kernelShape: number[], strides: number[], pads: number[], count_in_pad: number) : Tensor;

    add(a: Tensor, b: Tensor): Tensor;

    sub(a: Tensor, b: Tensor) : Tensor;

    logSumExp(x: Tensor) : Tensor;

    // return A * B * alpha + bias * beta, where A=(transposeA)?a':a, silimar to B
    // A => [M, K], B: [K, N], bias: castable[M, N]
    gemm(a: Tensor, b: Tensor, bias?: Tensor, alpha?: number, beta?: number, transposeA?: boolean, transposeB?: boolean) : Tensor;

    relu(x: Tensor) : Tensor;

    exp(x: Tensor): Tensor;

    // reduce
    sum(x: Tensor, axises?: number[], keepDims?: boolean) : Tensor;

    //treat [..., axis...] elements after given axis(including) vector, before axis as batchSize or sampleCount 
    softmax(logits: Tensor, axis?: number) : Tensor;

    reshape(x: Tensor, newShape: number[]) : Tensor;
    
    transpose(x: Tensor, perm?: number[]) : Tensor;
    
    matMul(a : Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor;

    neg(a: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;

    // slice(x: Tensor, begin: number[], size: number[]) : Tensor;
    // pad(x: Tensor, paddings: Array<[number, number]>, padValue: number): Tensor;
    tile(x: Tensor, repeats: number[]) : Tensor;
    pick(x: Tensor, indices: number[]) : Tensor;
}
