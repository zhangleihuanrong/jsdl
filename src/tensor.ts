import { DataType, Shape, BackendTensor, TensorLike, isTypedArray, getShape, StrictTensorLike } from './types';
import { ENV } from './environments';

export class Tensor {
    private static sNextId: number = 0;

    readonly id: number;
    data: BackendTensor;

    constructor(dtype: DataType, shape: number[], values: StrictTensorLike = null, backendTensor: BackendTensor = null) {
        this.id = Tensor.sNextId++;
        this.data = null;
        if (backendTensor) {
            ENV.engine.wrap(this, dtype, shape, backendTensor);
        }
        else {
            ENV.engine.make(this, dtype, shape, values);
        }
    }

    get shape() : Shape {
        return ENV.engine.backend.tensorShape(this);
    }

    get dtype(): DataType {
        return ENV.engine.backend.tensorDtype(this);
    }

    get size() : number {
        return ENV.engine.backend.tensorSize(this);
    }

    static create(values: TensorLike, shape: number[] = null, dtype: DataType = 'float32') : Tensor {
        const sa: StrictTensorLike = 
            (!isTypedArray(values) && !Array.isArray(values) && values != null) ? 
            ([values] as number[]) : 
            (values as StrictTensorLike);

        // TODO: Check shapes & types...
        shape = shape || getShape(sa);
        return new Tensor(dtype, shape, sa, null);
    }

    get rank() : number {
        return this.shape.length;
    }
}
