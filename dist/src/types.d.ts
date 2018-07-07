export interface DataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
}
export declare type DataType = keyof DataTypeMap;
export declare type TypedArray = DataTypeMap[DataType];
export declare type FlatVector = boolean[] | number[] | TypedArray;
export declare type RegularArray<T> = T[] | T[][] | T[][][] | T[][][][] | T[][][][][] | T[][][][][][];
export declare type ArrayData<D extends DataType> = DataTypeMap[D] | RegularArray<number> | RegularArray<boolean>;
export interface RecursiveArray<T extends any> {
    [index: number]: T | RecursiveArray<T>;
}
export declare type TensorLike = TypedArray | number | RegularArray<number> | boolean | RegularArray<boolean>;
export declare type BackendTensor = object;
export declare function isTypedArray(a: any): boolean;
export declare function getShape(val: any): number[];
export declare function toTypedArray<D extends DataType>(a: ArrayData<D>, dtype: D): DataTypeMap[D];
export declare function flatten<T extends number | boolean>(arr: T | RecursiveArray<T>, ret?: T[]): T[];
