
export interface ndvDataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
    uint8clamp: Uint8ClampedArray;
    string: string[];
    number: number[];
};

export type ndvDataType = keyof ndvDataTypeMap;
export type ndvArray = ndvDataTypeMap[ndvDataType];

export class NdView {
    dtype: ndvDataType;
    data: ndvArray;
    shape: number[];
    stride: number[];
    offset: number;

    readonly size: number;

    constructor(arr: ndvArray, shape:number[], dtype?: ndvDataType) {
        if (!shape || shape.some(v => v <= 0)) {
            throw new Error('Dimensions must be positive!');
        }

        this.data = arr || null;
        this.shape = shape;
        this.offset = 0;
        if (!arr) {
            this.dtype = dtype || 'float32';
        }
        else if (Array.isArray(arr)) {
            if (arr.length == 0) {
                this.dtype = 'number';
                this.data = null;
            }
            else {
                const fv: any = arr[0];
                if (fv instanceof String) {
                    this.dtype = 'string';
                }
                else if (fv instanceof Number) {
                    this.dtype = 'number';
                }
                else {
                    throw new Error('Wrong array type to construct!');
                }
            }
        }
        else if (arr instanceof Float32Array) {
            dtype = 'float32';
        }
        else if (arr instanceof Int32Array) {
            dtype = 'int32';
        }
        else if (arr instanceof Uint8Array) {
            dtype = 'bool';
        }
        else if (arr instanceof Uint8ClampedArray) {
            dtype = 'uint8clamp';
        }
        else {
            throw new Error('wrong array type to construct!')
        }

        let s = 1;
        this.stride.map(v => { const oldS = s;  s *= v;  return oldS; })
        this.size = this.shape.reduce((m, v) => m * v, 1);
    }

    index(...pos: number[]) : number {
        return this.stride.reduce((p, s, i) => p + s * pos[i], this.offset);
    }

    transpose(perm?: number[]) : NdView {
        const rank = this.shape.length;
        perm = perm || ((new Array(rank)).map((v, i) => rank - 1 - i));
        if (perm.length != rank) throw new Error('Wrong permutation size!');
        const chk = (new Array(rank)).fill(0);
        perm.forEach(v => {
            if (v >= 0 && v < rank) ++chk[v];
        });
        if (chk.some(v => v !== 1)) throw new Error('Wrong permutation!');
        if (chk.every((v, i) => v == i)) return this;

        const nshape = ((new Array(rank)).map((v, i) => this.shape[perm[i]]));
        const nstride = ((new Array(rank)).map((v, i) => this.stride[perm[i]]));
        const nv = new NdView(this.data, nshape, this.dtype);
        nv.stride = nstride;
        nv.offset = this.offset;
        return nv;
    }
    

};