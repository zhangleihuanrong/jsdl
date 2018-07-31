import { assert as ASSERT } from '../utils/gadget';

// N Dimention View structure on plain array.
// No data is stored or used here.
export type NDArrayLike = number[] | boolean[] | string[] | Float32Array | Int32Array | Uint8Array | Uint8ClampedArray | Float64Array;

// Need support:
// transpose in place,
// slice in place when no (repeat, gather, padding)
// gather in place*,
// padding in place*,
// repeat in place*
//
// pad in place works on core*, but not works together with other two: gather and repeat to simplify.
// 
// gather & repeat: 
//     possible status: 
//         gather(core), padding(core),
//         repeat(core), repeat(gather(core)), repeat(padding(core))
// So, if gather and repeat co-exists, apply gather first. padding + repeat means repeat(padding).
//
//     gather/padding/repeat operations:
//         gather1(gather2(core)) => gather'(core)
//         gather(repeat(core)) => gather'(core)
//         gather1(repeat(gather2(core)) => gather'(core)
//    
//         padding1(padding2(core)) => padding'(core)
//         
//         repeat1(repeat2(*)) => repeat'(*)
//         repeat(core/gather/padding) => repeat(*)
// 
// gather do not co-exists with padding. other operations should rebuild the data to get a clean
// core view first.
//
export class NDView<TARRAY extends NDArrayLike> {
    data: TARRAY = null;
    readonly coreShape: number[];
    readonly coreStride: number[];
    readonly coreOffset: number;

    readonly gather: number[][];
    readonly repeat: number[];
    readonly padding: [number, number][];
    readonly paddingValue: string | number;

    readonly needRepeat: boolean;
    readonly needGather: boolean;

    readonly coreSize: number; // size without considering the repeat
    readonly size: number;    // size after repeated
    readonly shape: number[]; // shape after repeat

    // data could be null
    constructor(
        data: TARRAY, 
        shape: number[],
        stride: number[] = null, 
        offset: number = 0,
        gather: number[][] = null,
        repeat: number[] = null,
        padding: [number, number][] = null,
        paddingValue : string | number = 0) {

        //TODO: slice(0) for all the non-null parameters except data
        const self = this;
        ASSERT(shape != null && shape.every(v => v > 0 && (v == (v|0))), 'Dimensions must all positive integers!');
        this.coreShape = shape.slice(0);
        const rank = this.coreShape.length;
        this.coreSize = this.coreShape.reduce((m, v) => m * v, 1);

        this.coreStride = stride ? stride.slice(0) : NDView.buildStride(shape);
        ASSERT(this.coreStride.length == rank, 'strides must of same length as shape!');
        ASSERT(this.coreStride.every(v => v == (v|0)), 'strides must of integers!');

        // shape after gather/padding
        let gprShape = this.coreShape.slice(0);

        this.gather = (gather) ? gather.slice(0) : null;
        if (this.gather != null) {
            ASSERT(this.gather.length == rank, "gather array should of same length as shape!");
            ASSERT(this.gather.every(a => a != null || Array.isArray(a)), 
                    "gather value for axis should all be number[]");
            ASSERT(this.gather.every(a => a.every((v, i) => v >= 0 && v == (v|0) && v < self.shape[i])),
                    `gather array value out of bound or not integer`);

            if (this.gather.some(a => a.length > 0) ) {
                gprShape = this.gather.map((a, i) => (a == null || a.length == 0) ? self.coreShape[i] : a.length);
            }
            else {
                this.gather = null; // in fact no gather
            }
        }

        this.padding = (padding) ? padding.slice(0) : null;
        ASSERT(!(this.padding != null && this.gather != null), "Padding can not co-existing with gather");
        if (this.padding != null) {
            ASSERT(this.padding.length == rank, "padding array should of same length as shape!");
            ASSERT(this.padding.every(v => v != null && Array.isArray(v) && v.length == 2), 
                        "Padding on axis should all be number[2]");
            ASSERT(this.padding.every(v => v[0] >= 0 && v[1] >= 0 && v[0] == (v[0]|0) && v[1] == (v[1]|0)), 
                        "Padding on axis should all zero or positive integers");
            if (this.padding.some(a => a[0] != 0 || a[1] != 0)) {
                gprShape = gprShape.map((c, idx) => c + self.padding[idx][0] + self.padding[idx][1]);
            }
            else {
                this.padding = null;
            }
        }

        this.repeat = repeat || null;
        if (this.repeat != null) {
            ASSERT(this.repeat.length == this.coreShape.length, "repeat must all of same dimentions as shape!");
            ASSERT(this.repeat.every(v => v > 0 && v == (v|0)), "repeat must all be positive integers!");
            //all repeat once means no repeat in fact
            if (!this.repeat.every(v => v == 1)) {
                gprShape = gprShape.map((w, i) =>  w * self.repeat[i]);
            }
            else {
                this.repeat = null;
            }
        }

        this.shape = gprShape;
        this.size = this.shape.reduce((m, v) => m * v, 1);
        
        this.coreOffset = offset;
        this.paddingValue = paddingValue;
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
        const coreWide = this.coreShape[axis];
        if (this.repeat) {
            let cgpWide = coreWide;
            if (this.padding) cgpWide += (this.padding[axis][0] + this.padding[axis][1]);
            if (this.gather && this.gather[axis].length > 0) cgpWide = this.gather[axis].length;
            outerIndex = outerIndex % cgpWide;
        }

        if (this.padding) {
            if (outerIndex >= this.padding[axis][0] && outerIndex < (this.padding[axis][0] + coreWide)) {
                return outerIndex - this.padding[axis][0];
            }
            return -1;
        }
        if (this.gather && this.gather[axis].length > 0) {
            return this.gather[axis][outerIndex];
        } 
        return outerIndex;
    }

    
    // return the index in the core flat array for given subscription array
    // no error check here
    index(...pos: number[]): number {
        const self = this;
        return this.coreStride.reduce((start, strideWide, axis) => {
            const index = this.coreIndexOnAxis(pos[axis], axis);
            return (start < 0 || index < 0)? -1 : start + strideWide * index;
        }, self.coreOffset);
    }

    get(...pos: number[]): any {
        const idx = this.index(...pos);
        return (idx >= 0) ? this.data[idx] : this.paddingValue;
    }

    set(pos: number[], value : any) {
        const idx = this.index(...pos);
        ASSERT(idx >= 0, "padding area are not allowed to set value");
        this.data[idx] = value;
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
        const nGather = (self.gather) ? self.gather.map((v, i, a) => a[perm[i]]) : null;
        const nRepeat = (self.repeat) ? self.repeat.map((v, i, a) => a[perm[i]]) : null;
        const nPadding = (self.padding) ? self.padding.map((v, i, a) => a[perm[i]]) : null;
        return new NDView(this.data, nshape, nstride, this.coreOffset, nGather, nRepeat, nPadding, this.paddingValue);
    }


    pick(indices: number[]): NDView<TARRAY> {
        const self = this;
        ASSERT(indices.length == this.coreShape.length, "pick should give indices of same length as shape!");
        ASSERT(indices.every((v, i) => v < self.shape[i]), "pick should use value less than axis' size or negitive for keep");
        const rank = this.coreShape.length;

        const nShape: number[] = [];
        const nStride: number[] = [];
        const nGather: number[][] = (self.gather) ? [] : null;
        const nRepeat: number[] = (self.repeat) ? [] : null;
        const nPadding: [number, number][] = (self.padding) ? []: null;

        let nOffset = self.coreOffset;
        for (let axis = 0; axis < rank; ++axis) {
            if (indices[axis] >= 0) {
                const ci = this.coreIndexOnAxis(indices[axis], axis);
                nOffset += this.coreStride[axis] * ci;
            }
            else {
                nShape.push(this.coreShape[axis]);
                nStride.push(this.coreStride[axis]);
                if (nGather != null) nGather.push(this.gather[axis]);
                if (nRepeat != null) nRepeat.push(this.repeat[axis]);
                if (nPadding != null) nPadding.push(this.padding[axis]);
            }
        }
        return new NDView(this.data, nShape, nStride, nOffset, nGather, nRepeat, nPadding, this.paddingValue);
    }


    squeeze(axises? : number[]) : NDView<TARRAY> {
        const self = this;
        const rank = self.coreShape.length;
        if (typeof axises === 'undefined' || axises == null || axises.length == 0) {
            axises = [];
            self.shape.forEach((v, index) => {
                if (v == 1) axises.push(index); 
            });
        }
        ASSERT(axises.every(v => v >= 0 && v <= rank), `squeeze dim parameter out of [0, ${rank}) -- ${axises}!`);
        ASSERT(axises.every(v => self.shape[v] == 1), `squeeze only allowed on axis of size 1. -- ${axises}`);

        // Sort ascedent and dedupe
        axises.sort(function(a, b){return a - b});
        for (let i = 0; i < axises.length - 1;) {
            if (axises[i] == axises[i+1]) {
                axises.splice(i, 1);
            }
            else {
                ++i;
            }
        }

        let eshape = this.coreShape.slice(0);
        const estride = this.coreStride.slice(0);  
        const egather = (this.gather != null) ? this.gather.slice(0) : null;
        const erepeat = (this.repeat) ? this.repeat.slice(0) : null;
        const epadding = (this.padding) ? this.padding.slice(0) : null;

        let removedCount = 0;
        for (let i = 0; i < axises.length; ++i) {
            const nposition = axises[i] - removedCount;
            ++removedCount;
            eshape.splice(nposition, 1);
            estride.splice(nposition, 1);
            if (egather) egather.splice(nposition, 1);
            if (erepeat) erepeat.splice(nposition, 1);
            if (epadding) epadding.splice(nposition, 1);
        }

        return new NDView(this.data, eshape, estride, this.coreOffset, egather, erepeat, epadding, this.paddingValue);
    }


    unsqueeze(axises : number[]) : NDView<TARRAY> {
        const rank = this.coreShape.length;

        axises = (axises == null) ? [0] : (axises.length == 0) ? [0] : axises;
        axises.sort(function(a, b){return a - b});
        ASSERT(axises.every(v => v >= 0 && v <= rank), "Expand dim parameter out of range!");

        let eshape = this.coreShape.slice(0);
        const estride = this.coreStride.slice(0);  
        const egather = (this.gather) ? this.gather.slice(0) : null;
        const erepeat = (this.repeat) ? this.repeat.slice(0) : null;
        const epadding = (this.padding) ? this.padding.slice(0) : null;

        let expand = axises.length - 1;
        for (let i = rank; i >= 0; --i) {
            for(; axises[expand] == i; --expand) {
                eshape.splice(i, 0, 1);
                estride.splice(i, 0, 0); // Add stride 0 element axises
                if (egather) egather.splice(i, 0, null);
                if (erepeat) erepeat.splice(i, 0, 1);
                if (epadding) epadding.splice(i, 0, [0, 0]);
            }
        }

        return new NDView(this.data, eshape, estride, this.coreOffset, egather, erepeat, epadding, this.paddingValue);
    }


    rebuildData(): TARRAY {
        if (this.data == null) return null;
        
        // Check originality
        if (this.padding == null && this.gather == null && this.repeat == null && this.size == this.coreSize) {
            if (this.coreOffset == 0 && this.size == this.data.length) {
                let isStrideDesc = true;
                for (let i = 0; i < this.coreShape.length - 1; ++i) {
                    if (this.coreStride[i] < this.coreStride[i+1]) isStrideDesc = false;
                }
                if (isStrideDesc) return this.data;
            }
        }

        let d: NDArrayLike = null;
        if (Array.isArray(this.data)) {
            d = new Array[this.size];
            this.forEach((v, idx) => d[idx] = v);
        }
        else if (this.data instanceof Float32Array){
            d = new Float32Array(this.size);
            this.forEach((v, idx) => d[idx] = v);
        }
        else if (this.data instanceof Int32Array) {
            d = new Int32Array(this.size);
            this.forEach((v, idx) => d[idx] = v);
        }
        else if (this.data instanceof Uint8Array) {
            d = new Uint8Array(this.size);
            this.forEach((v, idx) => d[idx] = v);
        }
        //TODO, more here
        return d as TARRAY;
    }


    rebuild(forceCopy: boolean = false) : NDView<TARRAY> {
        let pureCore: NDView<TARRAY> = this;
        if (this.padding || this.gather || this.repeat || forceCopy) {
            const arr = this.rebuildData();
            pureCore = new NDView(arr, this.shape, null, 0, null, null, null, 0);
        }
        return pureCore;
    }

    expandDim(axis?: number) : NDView<TARRAY> {
        axis = axis || 0;
        return this.unsqueeze([axis]);
    }


    reshape(shape: number[]): NDView<TARRAY> {
        ASSERT(shape.every(v => v > 0 || v == -1), "reshape axis len must be positive or -1");
        const numberOfNeg1 = shape.reduce((n, v) => n + ((v==-1)?1:0), 0);
        ASSERT(numberOfNeg1 <= 1, "At most one -1 could be used in reshape!");
        let ns = Math.abs(shape.reduce((m, v) => m*v, 1));
        if (numberOfNeg1 == 1) {
            ASSERT(this.size % ns == 0, "-1 can not find matching size during reshape");
            const w = Math.ceil(this.size / ns);
            shape = shape.map(v => (v == -1)? w : v);
            ns = ns * w;
        }
        ASSERT(ns == this.size, "Size not matching with original!");

        const arr = this.rebuildData();
        return new NDView(arr, shape, null, 0, null, null, null, 0);
    }


    pad(paddings: [number, number][], value: number|string = 0) : NDView<TARRAY> {
        ASSERT(paddings != null && paddings.length == this.shape.length, "Shape length not matching with padding dimentions");
        ASSERT(paddings.every(v => v != null && v[0] >= 0 && v[1] >= 0), "Padding on axis should all not negative.");
        if (this.gather || this.repeat) {
            const arr = this.rebuildData();
            return new NDView(arr, this.shape, null, 0, null, null, paddings, value);
        }
        
        if (this.padding) {
            paddings = paddings.map((v, i) => [v[0] + this.padding[i][0], v[1] + this.padding[i][1]]) as [number, number][];
        }
        return new NDView(this.data, this.coreShape, this.coreStride, this.coreOffset, null, null, paddings, value);
    }


    gatherOn(indices: number[], axis: number = 0) {
        ASSERT(axis >= 0 && axis < this.shape.length, "axis is out of bound");
        ASSERT((indices != null && indices.every(v => v >= 0 && v < this.shape.length && v == (v|0))), 
              "indices bay be negetive or out of bound or not integer!");

        let ngather: number[][] = this.shape.map((v, idx) => (idx == axis) ? indices : []);
        if (this.padding) {
            const core = this.rebuild();
            return new NDView(core.data, core.coreShape, core.coreStride, core.coreOffset, ngather, null, null, 0);
        }

        let nrepeat = (this.repeat) ? this.repeat.slice(0) : null;
        const innerGatherWide = (this.gather &&  this.gather[axis].length > 0) ? this.gather[axis].length : this.shape[axis];
        if (this.repeat && this.repeat[axis] != 1) {
            ngather[axis] = ngather[axis].map(v => v % innerGatherWide);
            nrepeat[axis] = 1;
        }
        
        if (this.gather &&  this.gather[axis].length > 0) {
            ngather = ngather.map((s, idx) => (idx != axis) ? this.gather[idx].slice(0) : s.map(v => this.gather[idx][v]));
        }

        return new NDView(this.data, this.coreShape, this.coreStride, this.coreOffset, ngather, nrepeat, null, 0);
    }

    tile(reps: number[]) : NDView<TARRAY> {
        ASSERT(reps != null && reps.length == this.shape.length, "title parameter length not matching shape");
        ASSERT(reps.every(v => v > 0 && v == (v|0)), "repeat must all be positive integers!");

        if (this.repeat) {
            reps = reps.map((v, idx) => (v * this.repeat[idx]));
        }
        return new NDView(this.data, this.coreShape, this.coreStride, this.coreOffset,
                          this.gather, reps, this.padding, this.paddingValue);
    }


    step(steps: number[]) : NDView<TARRAY> {
        ASSERT(steps && steps.length != this.shape.length && steps.every(v => (v|0) != 0),
               "steps must all be non-zero numbers!");
        steps = steps.map(v => v | 0); // to integer
        let pureCore = this.rebuild(false);

        let noffset = pureCore.coreOffset;
        const nshape = pureCore.shape.slice(0);
        const nstride = pureCore.coreStride.slice(0);
        for (let i = 0; i < nshape.length; ++i) {
            const d: number = steps[i];
            if (d < 0) {
                noffset += nstride[i] * (nshape[i] - 1);
            }
            nshape[i] = Math.ceil(nshape[i] / Math.abs(d));
            nstride[i] *= d;
        }
        return new NDView(pureCore.data, nshape, nstride, noffset, null, null, null, 0);
    }


    reverse(axis: number = 0) : NDView<TARRAY> {
        ASSERT(axis >= 0 && axis < this.shape.length, "axis out of boundary");
        const steps = this.shape.map((v, idx) => (idx == axis) ? -1 : 1);
        return this.step(steps);
    }

    // start should in [0, axis-len), size should be valid positive or -1.
    slice(start: number | number[], sizes?: number|number[]): NDView<TARRAY> {
        if (!sizes) sizes = this.shape.map(v => -1);
        if (!Array.isArray(start)) start = this.shape.map((v, idx) => (idx == 0) ? start as number : 0);
        if (!Array.isArray(sizes)) sizes = this.shape.map((v, idx) => (idx == 0) ? sizes as number : -1);
        sizes = sizes.map(v => v | 0);
        start = start.map(v => v | 0);
        ASSERT(start.length == this.shape.length && sizes.length == this.shape.length, "slice should use same length as shape");
        ASSERT(start.every((v, idx) => v >= 0 && v < this.shape[idx]), "starting location out of bound");
        ASSERT(sizes.every((s, idx) => (s == -1 || (s > 0 && s + start[idx] <= this.shape[idx]))), "sizes out of boundary");

        //lo
        const pureCore = this.rebuild(false);
        let noffset = pureCore.coreOffset;
        const nshape = pureCore.coreShape.slice(0);
        const nstride = pureCore.coreStride.slice(0);
        for (let i = 0; i < pureCore.shape.length; ++i) {
            noffset += start[i] * nstride[i];
            nshape[i] -= start[i];
        }

        // hi
        for (let i = 0; i < pureCore.shape.length; ++i) {
            if (sizes[i] > 0) nshape[i] = sizes[i];
        }

        return new NDView(pureCore.data, nshape, nstride, noffset, null, null, null, 0);
    }


    lo(...loc: number[]): NDView<TARRAY> {
        return this.slice(loc);
    }

    hi(...sizes: number[]): NDView<TARRAY> {
        return this.slice(0, sizes);
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
        printline: (line: string) => void = function (line) { console.log(line); },
        newLineAfterName: boolean = true
    ) {
        const shape = this.shape;
        const rank = shape.length;
        stringifyElem = (stringifyElem) ? stringifyElem : (x) => JSON.stringify(x);
        const excludes = this.getExcludes(rank, excludeLastAxis, excludeHiAxises);
        const loc = new Array(rank).fill(0);

        let currentline = [];
        if (name) {
            const dimensions: string = shape.join('x');
            currentline.push(`${name} ...${dimensions}... = `);
        }
        if (newLineAfterName && currentline.length > 0) {
            printline(currentline.join(''));
            currentline.length = 0;
        }

        const spacePrefix = new Array(rank + 1).fill('');
        if (currentline.length > 0) spacePrefix[0] = (new Array(currentline[0].length)).fill(' ').join('');
        for (let i = 1; i < rank; ++i) {
            spacePrefix[i] = `${spacePrefix[i-1]}  `; // two spaces
        }

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
        this.print('', null, [5,3], null, (line) => lines.push(line));
        return lines.join('\n');
    }

    
    recursiveEach(loc: number[], r: number, seqIdArr: [number],
                 cb: (v: any, index: number, loc: number[], arr: TARRAY) => void) {
        let limit = this.shape[r];
        if (r < this.shape.length-1) {
            for (loc[r] = 0; loc[r] < limit; ++loc[r]) {
                this.recursiveEach(loc, r+1, seqIdArr, cb);
            }
        }
        else {
            for (loc[r] = 0; loc[r] < limit; ++loc[r]) {
                const idx = this.index(...loc);
                const value = (idx >= 0) ? this.data[idx] : this.paddingValue;
                cb(value, seqIdArr[0], loc, this.data);
                ++seqIdArr[0];
            }
        }
    }


    forEach(cb: (v: any, seqId: number, location: number[]) => void) {
        const shape = this.shape;
        const rank = shape.length;
        const loc = new Array(rank).fill(0);
        const seqIdArr: [number] = [0];
        this.recursiveEach(loc, 0, seqIdArr, cb);
    }
};

