import {Tensor} from './tensor';

export interface Backend {
    matMul(a : Tensor, b: Tensor, transposeA: boolean, transposeB: boolean): Tensor;

    slice(x: Tensor, begin: number[], size: number[]) : Tensor;
    add(a: Tensor, b: Tensor): Tensor;
    neg(a: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;

    topKValues(x: Tensor, k: number) : Tensor;
    topKIndeces(x: Tensor, k: number) : Tensor;

    relu(x: Tensor) : Tensor;
    reshape(x: Tensor, shape: number[]) : Tensor;
    pad(x: Tensor, paddings: Array<[number, number]>, padValue: number): Tensor;
    transpose(x: Tensor, perm: number[]) : Tensor;
}
