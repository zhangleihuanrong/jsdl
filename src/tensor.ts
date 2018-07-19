import { DataType, Shape, BackendTensor, StrictTensorLike, getShape, TypedArray, toTypedArray, isTypeArrayFor, createTypeArrayForShape } from './types';
import { ENV } from './environments';
import { printNdarray } from './utils/ndarray_print';

import * as ndarray from 'ndarray';

export class Tensor {
    private static sNextId: number = 0;

    readonly id: number;
    data: BackendTensor;
    private _name: string = null;

    private constructor(values: TypedArray, shape: number[], dtype: DataType = 'float32') {
        this.id = Tensor.sNextId++;
        this.data = null;
        if (!values && !shape) {
            // Empty tensor. Will bind backend Tensor later by backend.
            return;
        }

        // TODO: check (shape vs values), (values vs dtype) if values provided.
        ENV.engine.make(this, dtype, shape, values);
    }

    static create(values: StrictTensorLike, shape: number[] = null, dtype: DataType = 'float32') : Tensor {
        if (values) {
            if (values instanceof Array) {
                const dataShape = getShape(values);
                const dataLength = dataShape.reduce((s, len) => s * len, 1);
                if (shape) {
                    // only check size
                    const length = shape.reduce((s, len) => s * len, 1);
                    if (length != dataLength) {
                        throw new Error(`Array shape:[${dataShape}] is not same as parameter [${shape}] requested!`);
                    }
                }
                else {
                    shape = dataShape;
                }
                const ta = toTypedArray(dtype, values);
                return new Tensor(ta, shape, dtype);
            }
            else {
                const valueSize = (values as TypedArray).length;
                if (!shape) {
                    shape = [valueSize];
                }
                const size = shape.reduce((s, len) => s * len, 1);
                if (size != (values as TypedArray).length) {
                    throw new Error(`Shape:${shape} not matching typed array length:${size}`);
                }
                if (!isTypeArrayFor(values, dtype)) {
                    throw new Error(`values in typed array not for type ${dtype}`);
                }
                return new Tensor(values as TypedArray, shape, dtype);
            }
        }
        if (!shape) return new Tensor(null, null);
        const ta = createTypeArrayForShape(dtype, shape);
        return new Tensor(ta, shape, dtype);
    }

    static fromBackend(backendTensor: BackendTensor) : Tensor {
        const t = new Tensor(null, null);
        ENV.engine.wrap(t, backendTensor);
        return t;
    }

    wrap(backendTensor: BackendTensor) : Tensor {
        if (this.data != null) throw new Error(`Tensor${this.id} not empty`);
        ENV.engine.wrap(this, backendTensor);
        return this;
    }

    get name(): string {
        return (this._name && this._name.length > 0) ? this._name : `Tensor${this.id}`;
    }

    set name(value: string) {
        this._name = value;
    }

    get shape() : Shape {
        return this.data.shape();
    }

    get dtype(): DataType {
        return this.data.dtype();
    }

    get size() : number {
        return this.data.size();
    }

    get rank() : number {
        return this.shape.length;
    }

    print(number2string: (x: number) => string = null, 
          excludeLastAxis: [number, number] = null,
          excludeHiAxises: [number, number] = null) {
        if (this.data) {
            const ta = ENV.engine.backend.readSync(this);
            const ndx = ndarray(ta, this.shape);
            printNdarray(ndx, this.name, number2string, excludeLastAxis, excludeHiAxises);
        }
        else {
            console.log(`${this.name} -- TensorId:${this.id} --:  contains no data`)
        }
    }
}
