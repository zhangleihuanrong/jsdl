import { TypedArray, DataType } from './types';
import { Tensor} from './tensor';
import { TensorManager } from './tensor_manager';

export interface Backend extends TensorManager{
    // Getting basic tensor properties
    tensorShape(t: Tensor): number[];
    tensorDtype(t: Tensor): DataType;
    tensorSize(t: Tensor): number;

    //init(t: Tensor, initializer: (t: Tensor)=>void);
    randomUniformEq(t: Tensor, a: number, b: number) : void;
    randomNormEq(t: Tensor, mean: number, variance: number, seed: number) : void;

    //write(t: Tensor, values: StrictTensorLike) : void;
    //read(t: Tensor) : Promise<TypedArray>;
    readSync(t: Tensor) : TypedArray;

    // in place random initializer. return the input tensor

    reshape(x: Tensor, newShape: number[]) : Tensor;
    transpose(x: Tensor, perm: number[]) : Tensor;
    matMul(a : Tensor, b: Tensor, transposeA?: boolean, transposeB?: boolean): Tensor;

    neg(a: Tensor): Tensor;
    add(a: Tensor, b: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;

    relu(x: Tensor) : Tensor;
    // reshape(x: Tensor, shape: number[]) : Tensor;

    // slice(x: Tensor, begin: number[], size: number[]) : Tensor;
    // pad(x: Tensor, paddings: Array<[number, number]>, padValue: number): Tensor;
}
