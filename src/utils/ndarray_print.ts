import * as ndarray from 'ndarray';

function printNdarrayRecursive(
    prefixes: string[],
    r: number, // depth of the array axises
    currentLine: string[], // elements in current line
    loc: number[], 
    shape: number[], 
    nda: ndarray,
    stringify: (number) => string, 
    excludes: [number, number][],
    printFunc: (line: string) => void
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
                printFunc(currentLine.join(''));
                currentLine.length = 0;
                currentLine.push(prefixes[r+1]);
                loc[r] = exRight - 1; 
            }
            else {
                printNdarrayRecursive(prefixes, r+1, currentLine, loc, shape, nda, stringify, excludes, printFunc);
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
                const v = nda.get(...loc);
                currentLine.push(stringify(v));
                if  (loc[r] != shape[r] - 1) {
                    currentLine.push(', ');
                }
            }
            ++loc[r];
        }
    }

    if (r > 0 && loc[r-1] < shape[r-1] - 1) {
        currentLine.push(' ],');
        printFunc(currentLine.join(''));
        currentLine.length = 0;
        currentLine.push(prefixes[r]);
    }
    else {
        currentLine.push(' ]');
        if (r == 0) {
            printFunc(currentLine.join(''));
            currentLine.length = 0;
        }
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
    excludeHiAxises: [number, number] = null,
    printFunc: (line: string) => void = function (line) { console.log(line); }
) {
    const shape = nda.shape;
    const rank = shape.length;
    const excludes = getExcludes(rank, excludeLastAxis, excludeHiAxises);
    const loc = new Array(rank).fill(0);
    const spacePrefix = new Array(rank).fill("");
    for (let i = 1; i < rank; ++i) {
        spacePrefix[i] = `${spacePrefix[i-1]}  `; // two spaces
    }
    
    console.log(`${name} of shape:${shape} = `);
    printNdarrayRecursive(spacePrefix, 0, [], loc, shape, nda, stringify, excludes, printFunc);
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

