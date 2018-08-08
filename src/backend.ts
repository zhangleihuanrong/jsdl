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

    reshape(x: Tensor, newShape: number[]) : Tensor;
    transpose(x: Tensor, perm?: number[]) : Tensor;
    matMul(a : Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor;

    neg(a: Tensor): Tensor;
    add(a: Tensor, b: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;

    relu(x: Tensor) : Tensor;
    // reshape(x: Tensor, shape: number[]) : Tensor;

    // slice(x: Tensor, begin: number[], size: number[]) : Tensor;
    // pad(x: Tensor, paddings: Array<[number, number]>, padValue: number): Tensor;
    tile(x: Tensor, repeats: number[]) : Tensor;
    pick(x: Tensor, indices: number[]) : Tensor;
}
