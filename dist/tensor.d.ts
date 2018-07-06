import { TensorLike, DataType } from "./types";
export declare class Tensor {
    id: number;
    shape: number[];
    size: number;
    dtype: DataType;
    constructor(shape: number[], dtype: DataType, values?: TensorLike);
}
