import { DataType, BackendTensor, TensorLike, StrictTensorLike } from './types';
export declare class Tensor {
    private static sNextId;
    id: number;
    shape: number[];
    dtype: DataType;
    backendTensor: BackendTensor;
    size: number;
    constructor(shape: number[], dtype: DataType, values?: StrictTensorLike, backendTensor?: BackendTensor);
    static create(values: TensorLike, shape?: number[], dtype?: DataType): Tensor;
    readonly rank: number;
}
