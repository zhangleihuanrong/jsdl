import { Tensor } from './tensor';
declare const transpose: (x: Tensor, perm: number[]) => Tensor;
declare const matMul: (a: Tensor, b: Tensor, transposeA: boolean, transposeB: boolean) => Tensor;
declare const neg: (a: Tensor) => Tensor;
declare const add: (a: Tensor, b: Tensor) => Tensor;
declare const multiply: (a: Tensor, b: Tensor) => Tensor;
declare const relu: (x: Tensor) => Tensor;
declare const tensor: typeof Tensor.create;
export { tensor, transpose, matMul, neg, add, multiply, relu };
