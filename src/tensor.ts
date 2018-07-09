import { DataType, BackendTensor, TensorLike, isTypedArray, getShape, StrictTensorLik, TypedArray } from './types';
import { ENV } from './environments';

export class Tensor {
    private static sNextId: number = 0;

    id: number;
    shape: number[];
    dtype: DataType;
    backendTensor: BackendTensor;

    size: number;


    protected constructor(shape: number[], dtype: DataType, values: StrictTensorLik = null, backendTensor: BackendTensor = null) {
        this.id = Tensor.sNextId++;
        this.shape = shape;
        this.dtype = dtype;
        this.size = (shape && shape.length > 0) ? shape.reduce ((a, b) => a*b, 1) : 0;
        this.backendTensor = backendTensor;
        ENV.engine.register(this);
        ENV.engine.backend.write(this, values);
    }

    static create(values: TensorLike, shape: number[] = null, dtype: DataType = 'float32') : Tensor {
        const sa: StrictTensorLik = 
            (!isTypedArray(values) && !Array.isArray(values)) ? 
            ([values] as number[]) : 
            (values as StrictTensorLik);

        // TODO: Check shapes & types...
        shape = shape || getShape(sa);
        return new Tensor(shape, dtype, sa);
    }

    get rank() : number {
        return this.shape.length;
    }
}
