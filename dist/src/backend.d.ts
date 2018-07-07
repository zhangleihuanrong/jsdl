import { BackendTensor, TypedArray } from './types';
import { Tensor } from './tensor';
import { TensorManager } from './tensor_manager';
export interface Backend extends TensorManager {
    write(bt: BackendTensor, values: TypedArray): void;
    read(bt: BackendTensor): Promise<TypedArray>;
    readSync(bt: BackendTensor): TypedArray;
    matMul(a: Tensor, b: Tensor, transposeA: boolean, transposeB: boolean): Tensor;
    transpose(x: Tensor, perm: number[]): Tensor;
    add(a: Tensor, b: Tensor): Tensor;
    neg(a: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;
    relu(x: Tensor): Tensor;
}
