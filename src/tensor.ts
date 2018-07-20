import { DataType, Shape, BackendTensor, StrictTensorLike, getShape, TypedArray, toTypedArray, isTypeArrayFor, createTypeArrayForShape } from './types';
import { ENV } from './environments';

import { printNdarray } from './utils/ndarray_print';
import { MPRandGauss } from './utils/rand';

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

    // Group of Creation methods
    // TODO: scala semantic is needed
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

    // TODO
    static scala(value: number|boolean, dtype?: DataType) {
        return null;
    }
    
    static randomUniform(shape: number[], minVal: number, maxVal: number, dtype: DataType = 'float32') : Tensor {
        const ta = createTypeArrayForShape(dtype, shape);
        for(let i = 0, limit = ta.length; i < limit; ++i) {
            ta[i] = Math.random() * (maxVal - minVal) + minVal;
        }
        return new Tensor(ta, shape, dtype);
    }

    static randomNorm(shape: number[], mean: number, stdDev: number, dtype: 'float32'|'int32' = 'float32', seed?: number) : Tensor {
        const randGauss = new MPRandGauss(mean, stdDev, dtype, false, seed);
        const ta = createTypeArrayForShape(dtype, shape);
        for(let i = 0, limit = ta.length; i < limit; ++i) {
            ta[i] = randGauss.nextValue();
        }
        return new Tensor(ta, shape, dtype);
    }
    
    static truncatedNorm(shape: number[], mean: number, stdDev: number, dtype: 'float32'|'int32' = 'float32', seed?: number) : Tensor {
        const randGauss = new MPRandGauss(mean, stdDev, dtype, true, seed);
        const ta = createTypeArrayForShape(dtype, shape);
        for(let i = 0, limit = ta.length; i < limit; ++i) {
            ta[i] = randGauss.nextValue();
        }
        return new Tensor(ta, shape, dtype);
    }

    static range(start: number, stop: number, step?:number, dtype?:DataType): Tensor {
        const len = Math.floor((stop - start)/step);
        const ta = createTypeArrayForShape(dtype, [len]);
        for(let i = 0, limit = ta.length; i < limit; i++) {
            ta[i] = start;
            start += step;
        }
        return new Tensor(ta, null, dtype);
    }

    static oneHot(indices: Tensor|Int32Array|number[], depth: number = -1, onValue: number = 1, offValue: number = 0): Tensor {
        if (!indices) {
            if (depth <= 0) throw new Error(`Depth should be positive integer rather than ${depth} when no indices data given.`);
            indices = Tensor.range(0, depth, 1, 'int32');
        }
        if (indices instanceof Tensor) {
            if (indices.shape.length != 1 || indices.dtype != 'int32') {
                throw new Error(`indices should be of 1D of int32 rather than ${indices.shape.length}D of ${indices.dtype}!`);
            } 
            indices = ENV.engine.backend.readSync(indices) as Int32Array;
        }
        if (indices instanceof Int32Array) {
            indices = Array.prototype.slice.call(indices) as number[];
        }
        if (indices.length > depth) {
            throw new Error(`Indices len:${indices.length} should smaller or equal than depth:${depth}`);
        }
        if (!indices.every(value => value < depth && value >= 0)) {
            throw new Error(`Indices value should all in [0, ${depth}]`);
        }
        if (depth <= 0) depth = indices.length;
        const shape: number[] = [indices.length, depth];
        const ta = createTypeArrayForShape('int32', shape);
        for (let row = 0; row < indices.length; row++) {
            for (let col = 0; col < depth; col++) {
                ta[row * depth + col] = (col ==indices[row]) ? onValue: offValue;
            }
        }
        return new Tensor(ta, shape);
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
