export type Shape = number[];

export interface DataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
}

export type DataType = keyof DataTypeMap;
export type TypedArray = DataTypeMap[DataType];
export type FlatVector = boolean[] | number[] | TypedArray;
export type RegularArray<T> = T[] | T[][] | T[][][] | T[][][][] | T[][][][][] | T[][][][][][];
export type ArrayData<D extends DataType> = DataTypeMap[D] | RegularArray<number> | RegularArray<boolean>;
export interface RecursiveArray<T extends any> {
    [index: number]: T | RecursiveArray<T>;
}

export type StrictTensorLike = TypedArray | RecursiveArray<number> | RecursiveArray<boolean>;
export type TensorLike = StrictTensorLike | number | boolean;

export type BackendTensor = object;

export function isTypedArray(a: any) : boolean {
    return a instanceof Float32Array || a instanceof Int32Array || a instanceof Uint8Array;
}

export function getShape(val: any) : number[] {
    if (isTypedArray(val)) { 
        return [(val as TypedArray).length]
    }
    if (!Array.isArray(val)) {
        return []; // scala
    }
    const shape: number[] = [];
    while (val instanceof Array) {
        shape.push(val.length);
        val = val[0];
    }
    return shape;
}

const ArrayDataInstances : object = {
    float32: new Float32Array([1.0]), 
    int32: new Int32Array([1]),
    bool: new Uint8Array([0]),
};
    
export function getTypedArraySample(dtype: DataType) : TypedArray {
    return ArrayDataInstances[dtype];
}

export function toTypedArray<D extends DataType>(a: ArrayData<D>, dtype: D) : DataTypeMap[D] {
    if ((a instanceof Float32Array && dtype === 'float32') ||
        (a instanceof Int32Array && dtype === 'int32') ||
        (a instanceof Uint8Array && dtype === 'bool')) {
            return a as DataTypeMap[D];
    }
    if (Array.isArray(a)) {
        const arr = flatten(a as number[]);
        if (dtype === 'float32' || dtype === 'int32') {
            return new Float32Array(arr as number[]);
        }
        else if (dtype === 'bool') {
            const bools = new Uint8Array(arr.length);
            for (let i = 0; i < bools.length; ++i) {
                bools[i] = (Math.round(arr[i] as number) !== 0) ? 1 : 0;
            }
            return bools;
        }
    }
    throw new Error('should not arrived here!')
}


export function flatten<T extends number|boolean>(arr: T | RecursiveArray<T>, ret: T[] = []) : T[] {
    if (Array.isArray(arr)) {
        for (let i = 0; i < arr.length; ++i) {
            flatten(arr[i], ret);
        }
    }
    else {
        ret.push(arr as T);
    }
    return ret;
}