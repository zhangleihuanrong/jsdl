import { TypedArray, StrictTensorLike } from './types';
import { Tensor } from './tensor';
import { TensorManager } from './tensor_manager';
export interface Backend extends TensorManager {
    write(t: Tensor, values: StrictTensorLike): void;
    read(t: Tensor): Promise<TypedArray>;
    readSync(t: Tensor): TypedArray;
    transpose(x: Tensor, perm: number[]): Tensor;
    matMul(a: Tensor, b: Tensor, transposeA: boolean, transposeB: boolean): Tensor;
    neg(a: Tensor): Tensor;
    add(a: Tensor, b: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;
    relu(x: Tensor): Tensor;
}
