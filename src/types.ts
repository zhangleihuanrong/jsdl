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

//export type BackendTensor = object;
export interface BackendTensor {
    shape(): number[];
    dtype(): DataType;
    size(): number;
}

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

export function shape2Size(shape: number[]) {
    return shape.reduce((m, v) => m * v, 1.0);
}

export function createTypedArray<D extends DataType>(dtype: D, len: number) : DataTypeMap[D] {
    if (dtype === 'float32') {
        return new Float32Array(len);
    }
    else if (dtype === 'int32') {
        return new Int32Array(len);
    }
    else if (dtype === 'bool') {
        return new Uint8Array(len);
    }

    throw new Error('Unsupported type:' + dtype);
}

export function createTypeArrayForShape<D extends DataType>(dtype: D, shape: number[]) : DataTypeMap[D] {
    const len = shape.reduce((m, v) => m * v, 1);
    return createTypedArray(dtype, len);
}

export function isTypeArrayFor(a: any, dtype: DataType) {
    return (a instanceof Float32Array && dtype === 'float32') ||
            (a instanceof Int32Array && dtype === 'int32') ||
            (a instanceof Uint8Array && dtype === 'bool');
}

export function toTypedArray<D extends DataType>(dtype: D, a: StrictTensorLike) : DataTypeMap[D] {
    if (isTypeArrayFor(a, dtype)) {
        return a as DataTypeMap[D];
    }
    if (Array.isArray(a)) {
        if (dtype === 'float32'){
            return new Float32Array(a as number[]);
        } else if (dtype === 'int32') {
            return new Int32Array(a as number[]);
        }
        else if (dtype === 'bool') {
            const shape = getShape(a);
            const bools = new Uint8Array(shape.reduce((s, len) => s*len, 1));
            const assigning = { index: 0 };
            arrayIterate(a, (x: any) => {
                bools[assigning.index] = x;
            });
            return bools;
        }
    }
    throw new Error('should not arrived here!')
}

export function arrayIterate(arr:any, cb: (x: any) => void) {
    if (Array.isArray(arr)) {
        for (let i = 0; i < arr.length; ++i) {
            arrayIterate(arr[i], cb);
        }
    }
    else {
        cb(arr);
    }
}