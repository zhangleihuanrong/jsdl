import * as ndarray from 'ndarray';

function printNdarrayRecursive(
    prefixes: string[],
    r: number, 
    loc: number[], 
    shape: number[], 
    nda: ndarray,
    stringify: (number) => string, 
    excludes: [number, number][]
) {
    let exRight = (excludes[r][1] >= 0) ? (excludes[r][1]) : (shape[r] + excludes[r][1]);
    if (r != shape.length-1) {
        console.log(`${prefixes[r]}[`);

        for (loc[r] = 0; loc[r] < shape[r]; ) {
            if (excludes && loc[r] == excludes[r][0] && loc[r] < exRight) {
                console.log(`${prefixes[r+1]}...`);
                loc[r] = exRight - 1; 
            }
            else {
                printNdarrayRecursive(prefixes, r+1, loc, shape, nda, stringify, excludes);
            }
            ++loc[r];
        }
 
        const tailComma = (r > 0 && loc[r-1] != shape[r-1] - 1) ? "," : "";
        console.log(`${prefixes[r]}]${tailComma}`);
    }
    else {
        let line = prefixes[r] + '[';
        for (loc[r] = 0; loc[r] < shape[r]; ) {
            if (excludes && loc[r] == excludes[r][0] && loc[r] < exRight) {
                line += ' ... ';
                loc[r] = exRight; 
            }
            else {
                const v = nda.get(...loc);
                line += stringify(v);
                if  (loc[r] != shape[r] - 1) {
                    line += ', ';
                }
            }
            ++loc[r];
        }
        const tailComma = (r > 0 && loc[r-1] != shape[r-1] - 1) ? "," : "";
        console.log(`${line}]${tailComma}`);
    }
}


function getExcludes(rank: number, excludeLastAxis: [number, number], excludeHiAxises: [number, number]): [number, number][] {
    excludeLastAxis = (excludeLastAxis)? excludeLastAxis : [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER];
    excludeHiAxises = (excludeHiAxises)? excludeHiAxises : excludeLastAxis;

    const excludes : [number, number][] = [];
    for (let i = 0; i < rank - 1; ++i) {
        excludes.push(excludeHiAxises);
    }
    excludes.push(excludeLastAxis);
    return excludes;
}


export function printNdarray(
    nda: ndarray,          
    name: string = '',
    stringify: (number) => string = (x:number) => x.toString(),
    excludeLastAxis: [number, number] = null,
    excludeHiAxises: [number, number] = null
) {
    const shape = nda.shape;
    const rank = shape.length;
    const excludes = getExcludes(rank, excludeLastAxis, excludeHiAxises);
    const loc = new Array(rank).fill(0);
    const spacePrefix = new Array(rank).fill("");
    for (let i = 1; i < rank; ++i) {
        spacePrefix[i] = `${spacePrefix[i-1]}  `;
    }
    
    console.log(`${name} of shape:${shape} = `);
    printNdarrayRecursive(spacePrefix, 0, loc, shape, nda, stringify, excludes);
}


function iterateNdarrayRecursive(nda: ndarray, loc: number[], r: number, cb: (arr: ndarray, loc: number[]) => void) {
    let limit = nda.shape[r];
    if (r < nda.shape.length-1) {
        for (loc[r] = 0; loc[r] < limit; ++loc[r]) {
            iterateNdarrayRecursive(nda, loc, r+1, cb);
        }
    }
    else {
        for (loc[r] = 0; loc[r] < limit; ++loc[r]) {
            cb(nda, loc);
        }
    }
}

export function iterateNdarray(nda: ndarray, cb: (arr: ndarray, location: number[]) => void) {
    const shape = nda.shape;
    const rank = shape.length;
    const loc = new Array(rank).fill(0);
    iterateNdarrayRecursive(nda, loc, 0, cb);
}

