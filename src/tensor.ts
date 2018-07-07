import { DataType, TypedArray, BackendTensor, TensorLike, isTypedArray, getShape, toTypedArray, FlatVector } from './types';
import { ENV } from './environments';
import * as ndarray from 'ndarray';

export class Tensor {
    private static sNextId: number = 0;

    id: number;
    shape: number[];
    dtype: DataType;
    tensor: ndarray;
    backendTensor: BackendTensor;

    protected constructor(shape: number[], dtype: DataType, values: TypedArray = null, backendTensor: BackendTensor = null) {
        this.id = Tensor.sNextId++;
        this.shape = shape;
        this.dtype = dtype;
        if (values != null) {
            this.tensor = ndarray(values, shape);
        }
        else {
            this.tensor = null; // TODO
        }
        this.backendTensor = backendTensor;
        ENV.engine.register(this);
        ENV.engine.backend.write(backendTensor, values);
    }

    get rank() : number {
        return this.shape.length;
    }

    static create(values: TensorLike, shape: number[] = null, dtype: DataType = 'float32') : Tensor {
        const fa: FlatVector = (!isTypedArray(values) && !Array.isArray(values)) ? 
            ([values] as number[]): (values as FlatVector);

        // TODO: Check shapes & types
        shape = shape || getShape(fa);
        const ta = toTypedArray(fa, dtype)
        return new Tensor(shape, dtype, ta);
    }

}