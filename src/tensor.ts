import { DataType, Shape, BackendTensor, TensorLike, isTypedArray, getShape, StrictTensorLike } from './types';
import { engine, backend } from './environments';

export class Tensor {
    private static sNextId: number = 0;

    readonly id: number;
    data: BackendTensor;

    constructor(dtype: DataType, shape: number[], values: StrictTensorLike = null, backendTensor: BackendTensor = null) {
        this.id = Tensor.sNextId++;
        if (backendTensor) {
            engine.wrap(this, dtype, shape, backendTensor);
        }
        else {
            engine.make(this, dtype, shape, values);
        }
    }

    get shape() : Shape {
        return backend.tensorShape(this);
    }

    get dtype(): DataType {
        return backend.tensorDtype(this);
    }

    get size() : number {
        return backend.tensorSize(this);
    }

    static create(values: TensorLike, shape: number[] = null, dtype: DataType = 'float32') : Tensor {
        const sa: StrictTensorLike = 
            (!isTypedArray(values) && !Array.isArray(values)) ? 
            ([values] as number[]) : 
            (values as StrictTensorLike);

        // TODO: Check shapes & types...
        shape = shape || getShape(sa);
        return new Tensor(dtype, shape, dtype, sa);
    }

    get rank() : number {
        return this.shape.length;
    }
}
