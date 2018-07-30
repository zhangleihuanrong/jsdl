import { assert as ASSERT } from '../utils/gadget';

// N Dimention View structure on plain array.
// No data is stored or used here.
export type NDArrayLike = number[] | boolean[] | string[] | Float32Array | Int32Array | Uint8Array | Uint8ClampedArray | Float64Array;

// Need support:
// transpose in place,
// slice in place when no repeat nor gather,
// gather in place*,
// repeat in place*
//
// gather & repeat: 
//     possible status: 
//         gather(core), 
//         repeat(core), 
//         repeat(gather(core))
//     other operations:
//         gather1(gather2(core)) => gather'(core)
//         gather(repeat(core)) => gather'(core)
//         gather1(repeat(gather2(core)) => gather'(core)
//         repeat1(repeat2(*)) => repeat'(*)
// So, if gather and repeat co-exists, apply gather first.
//
export class NDView<TARRAY extends NDArrayLike> {
    data: TARRAY = null;
    readonly coreShape: number[];
    readonly coreStride: number[];
    readonly coreOffset: number;

    readonly gather: number[][];
    readonly repeat: number[];

    readonly needRepeat: boolean;
    readonly needGather: boolean;

    readonly coreSize: number; // size without considering the repeat
    readonly size: number;    // size after repeated
    readonly shape: number[]; // shape after repeat

    // data could be null
    constructor(data: TARRAY, shape: number[],
        stride: number[] = null, offset: number = 0,
        gather: number[][] = null, repeat: number[] = null) {
        const self = this;

        ASSERT(shape != null && shape.every(v => v > 0), 'Dimensions must not null and all positive!');
        this.coreShape = shape;
        const rank = shape.length;
        this.coreSize = shape.reduce((m, v) => m * v, 1);

        this.coreStride = stride || NDView.buildStride(shape);
        ASSERT(this.coreStride.length == rank, 'strides must of same length as shape!');

        this.gather = gather || new Array(rank).fill(null);
        ASSERT(this.gather.length == rank, "gather array should of same length as shape!");
        ASSERT(this.gather.every(a => a == null || Array.isArray(a)), "gather should be number[][]");
        ASSERT(this.gather.every(a => a == null || a.every((v, i) => v >= 0 && v < self.shape[i])),
            `gather array value out of bound`);
        this.needGather = this.gather.some(a => a != null && a.length > 0);
        const gsa = this.gather.map((a, i) => (a == null || a.length == 0) ? self.coreShape[i] : a.length);

        this.repeat = repeat || new Array(rank).fill(1);
        ASSERT(this.repeat.every(v => v > 0), "repeat must all be positive!");
        this.needRepeat = this.repeat.some(v => v != 1);

        this.shape = gsa.map((w, i) => w * self.repeat[i]);
        this.size = this.shape.reduce((m, v) => m * v, 1);
        
        this.coreOffset = offset;
        this.data = data;
    }

    static buildStride(shape: number[]) : number[] {
        const stride: number[] = shape.map(v => 1);
        for (let i = shape.length - 2; i >= 0; --i) {
            stride[i] = shape[i+1] * stride[i+1];
        }
        return stride;
    }

    // no error check for performance
    private coreIndexOnAxis(outerIndex: number, axis: number) : number {
        const gatherOnAxis = (this.gather[axis] != null && this.gather[axis].length >= 0);
        const gatherWide =  gatherOnAxis ? this.gather[axis].length : this.coreShape[axis];
        let index = outerIndex % gatherWide;
        if (gatherOnAxis) index = this.gather[axis][index];
        return index;
     }

    // return the index in the core flat array for given subscription array
    // no error check here
    index(...pos: number[]): number {
        const self = this;
        return this.coreStride.reduce((start, strideWide, axis) => {
            const index = this.coreIndexOnAxis(pos[axis], axis);
            return start + strideWide * index;
        }, self.coreOffset);
    }

    get(...pos: number[]): any {
        const idx = this.index(...pos);
        return this.data[idx];
    }

    transpose(perm?: number[]): NDView<TARRAY> {
        const self = this;
        const rank = this.coreShape.length;
        perm = perm || ((new Array(rank)).map((v, i) => rank - 1 - i));
        ASSERT(perm.length == rank, 'Wrong permutation size!');

        const check = new Array(rank).fill(0);
        perm.forEach(v => { if (v >= 0 && v < rank)++(check[v]); });
        ASSERT(check.every(v => v === 1), 'Wrong permutation!');
        if (check.every((v, i) => v == i)) return this;

        const nshape = self.coreShape.map((v, i, a) => a[perm[i]]);
        const nstride = self.coreStride.map((v, i, a) => a[perm[i]]);
        const nGather = this.gather.map((v, i, a) => a[perm[i]]);
        const nRepeat = self.repeat.map((v, i, a) => a[perm[i]]);
        const nv = new NDView(this.data, nshape, nstride, this.coreOffset, nGather, nRepeat);
        return nv;
    }

    pick(indices: number[]): NDView<TARRAY> {
        const self = this;
        ASSERT(indices.length == this.coreShape.length, "pick should give indices of same length as shape!");
        ASSERT(indices.every((v, i) => v < self.shape[i]), "pick should use value less than axis' size or negitive for keep");
        const rank = this.coreShape.length;

        const nShape: number[] = [];
        const nStride: number[] = [];
        const nGather: number[][] = [];
        const nRepeat: number[] = [];

        let nOffset = self.coreOffset;
        for (let axis = 0; axis < rank; ++axis) {
            if (indices[axis] >= 0) {
                const ci = this.coreIndexOnAxis(indices[axis], axis);
                nOffset += this.coreStride[axis] * ci;
            }
            else {
                nShape.push(this.coreShape[axis]);
                nStride.push(this.coreStride[axis]);
                nGather.push(this.gather[axis]);
                nRepeat.push(this.repeat[axis]);
            }
        }
        return new NDView(this.data, nShape, nStride, nOffset, nGather, nRepeat);
    }

    printRecursively(
        prefixes: string[],
        r: number, // depth of the array axises
        currentLine: string[], // elements in current line
        loc: number[], 
        shape: number[], 
        excludes: [number, number][],
        stringifyElem: (any) => string, 
        printLine: (line: string) => void
    ) {
        let exRight = (excludes[r][1] >= 0) ? (excludes[r][1]) : (shape[r] + excludes[r][1]);
        currentLine.push('[ ');
        if (r != shape.length-1) {
            for (loc[r] = 0; loc[r] < shape[r]; ) {
                if (excludes && loc[r] == excludes[r][0] && loc[r] < exRight) {
                    for (let k = r+1; k < shape.length; ++k) currentLine.push('[ ');
                    currentLine.push(' ...... ');
                    for (let k = r+1; k < shape.length; ++k) currentLine.push(' ]...');
                    currentLine.push(',');
                    printLine(currentLine.join(''));
                    currentLine.length = 0;
                    currentLine.push(prefixes[r+1]);
                    loc[r] = exRight - 1; 
                }
                else {
                    this.printRecursively(prefixes, r+1, currentLine, loc, shape, excludes, stringifyElem, printLine);
                }
                ++loc[r];
            }
    
        }
        else {
            for (loc[r] = 0; loc[r] < shape[r]; ) {
                if (excludes && loc[r] == excludes[r][0] && loc[r] < exRight) {
                    currentLine.push(' ......, ');
                    loc[r] = exRight - 1; 
                }
                else {
                    const v = this.get(...loc);
                    currentLine.push(stringifyElem(v));
                    if  (loc[r] != shape[r] - 1) {
                        currentLine.push(', ');
                    }
                }
                ++loc[r];
            }
        }
    
        if (r > 0 && loc[r-1] < shape[r-1] - 1) {
            currentLine.push(' ],');
            printLine(currentLine.join(''));
            currentLine.length = 0;
            currentLine.push(prefixes[r]);
        }
        else {
            currentLine.push(' ]');
            if (r == 0) {
                printLine(currentLine.join(''));
                currentLine.length = 0;
            }
        }
    }
    

    getExcludes(rank: number, excludeLastAxis: [number, number], excludeHiAxises: [number, number]): [number, number][] {
        excludeLastAxis = (excludeLastAxis)? excludeLastAxis : [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER];
        excludeHiAxises = (excludeHiAxises)? excludeHiAxises : excludeLastAxis;
    
        const excludes : [number, number][] = [];
        for (let i = 0; i < rank - 1; ++i) {
            excludes.push(excludeHiAxises);
        }
        excludes.push(excludeLastAxis);
        return excludes;
    }
    
    
    print(
        name: string = '',
        stringifyElem: (any) => string = null, 
        excludeLastAxis: [number, number] = null,
        excludeHiAxises: [number, number] = null,
        leftMargin: number = 2,
        printline: (line: string) => void = function (line) { console.log(line); }
    ) {
        const shape = this.shape;
        const rank = shape.length;
        stringifyElem = (stringifyElem) ? stringifyElem : (x) => JSON.stringify(x);
        const excludes = this.getExcludes(rank, excludeLastAxis, excludeHiAxises);
        const loc = new Array(rank).fill(0);
        const spacePrefix = new Array(rank+1).fill('');
        if (rank >= 1) spacePrefix[0] = (new Array(leftMargin+1)).fill(' ').join('').toString();
        for (let i = 1; i < rank; ++i) {
            spacePrefix[i] = `${spacePrefix[i-1]}  `; // two spaces
        }
        
        console.log(`${name} of shape:${shape} = `);
        const currentline = [spacePrefix[0]];
        if (rank <= 0) {
            const s = stringifyElem(this.data[0]);
            currentline.push(s);
            printline(currentline.join(''));
            currentline.length = 0;
        }
        else {
            this.printRecursively(spacePrefix, 0, currentline, loc, shape, excludes, stringifyElem, printline);
        }
    }

    toString( ) {
        const lines: string[] = [];
        this.print('', null, [5,3], null, 0, (line) => lines.push(line));
        return lines.join('\n');
    }
    
};

