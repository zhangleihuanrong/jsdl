import { BackendTensor, TypedArray, StrictTensorLik } from './types';
import { Tensor} from './tensor';
import { TensorManager } from './tensor_manager';

export interface Backend extends TensorManager{
    write(t: Tensor, values: StrictTensorLik) : void;
    read(t: Tensor) : Promise<TypedArray>;
    readSync(t: Tensor) : TypedArray;

    matMul(a : Tensor, b: Tensor, transposeA: boolean, transposeB: boolean): Tensor;
    transpose(x: Tensor, perm: number[]) : Tensor;

    add(a: Tensor, b: Tensor): Tensor;
    neg(a: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;

    relu(x: Tensor) : Tensor;
    // reshape(x: Tensor, shape: number[]) : Tensor;

    // slice(x: Tensor, begin: number[], size: number[]) : Tensor;
    // pad(x: Tensor, paddings: Array<[number, number]>, padValue: number): Tensor;
}
