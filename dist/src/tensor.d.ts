import { DataType, TypedArray, BackendTensor, TensorLike } from './types';
import * as ndarray from 'ndarray';
export declare class Tensor {
    private static sNextId;
    id: number;
    shape: number[];
    dtype: DataType;
    tensor: ndarray;
    backendTensor: BackendTensor;
    protected constructor(shape: number[], dtype: DataType, values?: TypedArray, backendTensor?: BackendTensor);
    readonly rank: number;
    static create(values: TensorLike, shape?: number[], dtype?: DataType): Tensor;
}
