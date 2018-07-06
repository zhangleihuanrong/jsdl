import ndarray from 'ndarray';

import { DataType, TensorLike } from './types';


export class Tensor {
    //private static nextId = 0;
    id: number;
    shape: number[];
    size: number;
    dtype: DataType;
    values: any; // ndarray

    constructor(shape?: number[], dtype?: DataType, values?: TensorLike) {
        this.shape = shape;
        this.values = ndarray(values, shape);
    }
}