import ndarray from 'ndarray';
export declare type RegularArray<T> = T[] | T[][] | T[][][] | T[][][][] | T[][][][][] | T[][][][][][];
export interface DataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
}
export declare type DataType = keyof DataTypeMap;
export declare type TypedArray = DataTypeMap[DataType];
export declare type TensorLike = ndarray | TypedArray | number | RegularArray<number> | boolean | RegularArray<boolean>;
