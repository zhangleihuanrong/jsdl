
export type RegularArray<T> = T[] | T[][] | T[][][] | T[][][][] | T[][][][][] | T[][][][][][];

export interface DataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
}

export type DataType = keyof DataTypeMap;
export type TypedArray = DataTypeMap[DataType];

export type TensorLike = TypedArray | number | RegularArray<number> | boolean | RegularArray<boolean>;
