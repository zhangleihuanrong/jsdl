import { DataType, TypedArray, BackendTensor, TensorLike, isTypedArray, getShape, toTypedArray, FlatVector } from './types';


export class Tensor {
    private static sNextId: number = 0;

    id: number;
    shape: number[];
    size: number;
    dtype: DataType;

    protected constructor(shape: number[], dtype: DataType, values?: TypedArray, backendTensor?: BackendTensor) {
        this.id = Tensor.sNextId++;
        this.shape = shape;
        this.dtype = dtype;
    }

    get rank() : number {
        return this.shape.length;
    }

    static create(values: TensorLike, shape?: number[], dtype?: DataType = 'float32') : Tensor {
        const fa: FlatVector = (!isTypedArray(values) && !Array.isArray(values)) ? 
            ([values] as number[]): (values as FlatVector);

        // TODO: Check shapes & types
        shape = shape || getShape(fa);
        const ta = toTypedArray(fa, dtype)
        return new Tensor(shape, dtype, ta);
    }

}