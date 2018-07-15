import { TypedArray, DataType } from './types';
import { Tensor} from './tensor';
import { TensorManager } from './tensor_manager';

export class TensorPrintOptions {
    readonly stringify: (x: number) => string;
    readonly excludeLastAxis: [number, number];
    readonly excludeHiAxises: [number, number];

    constructor(stringify : (x: number) => string = null, 
                excludeLastAxis: [number, number] = null, 
                excludeHiAxises: [number, number] = null) {
        this.stringify = stringify ? stringify : x => x.toString(); 
        this.excludeLastAxis = (excludeLastAxis) ? excludeLastAxis : [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER];
        this.excludeHiAxises = (excludeHiAxises) ? excludeHiAxises : this.excludeLastAxis;
    }
};

export interface Backend extends TensorManager{
    // Getting basic tensor properties
    tensorShape(t: Tensor): number[];
    tensorDtype(t: Tensor): DataType;
    tensorSize(t: Tensor): number;

    print(t: Tensor, options?: TensorPrintOptions);

    //init(t: Tensor, initializer: (t: Tensor)=>void);
    randomUniformEq(t: Tensor, a: number, b: number) : void;
    randomNormEq(t: Tensor, mean: number, variance: number, seed: number) : void;

    //write(t: Tensor, values: StrictTensorLike) : void;
    //read(t: Tensor) : Promise<TypedArray>;
    readSync(t: Tensor) : TypedArray;

    // in place random initializer. return the input tensor

    conv2d(x: Tensor, filter: Tensor, strides: number | [number, number], padding: number[], dataFormat: 'NHWC' | 'NCHW', dialations: number | [number, number]) : Tensor;
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
