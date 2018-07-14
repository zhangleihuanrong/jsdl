import { Tensor } from './tensor';
import * as tf from './ops';

import * as ndarray from 'ndarray';

function ndarray_print2d(arr, name: string = '', heading: number, tailing: number) {
    console.log(`  ==shape for array ${name} is ${arr.shape}`)
    for (let y = 0; y < arr.shape[0]; ) {
        let line = "";
        if (y >= heading && y < arr.shape[0] - tailing) {
            line = " ... ";
            y = arr.shape[0] - tailing;
        }
        else {
            for (let x = 0; x < arr.shape[1]; ++x) {
                line += " " + arr.get(y, x) ;
            }
            ++y;
        }
        console.log("   ", line);
    }
}

export function print(t: Tensor, name: string = '', heading: number = 5, tailing: number = 3) {
    const ta = tf.readSync(t);
    const arr = ndarray(ta, t.shape);

    ndarray_print2d(arr, name, heading, tailing);
}

export function printNdarrayRecursive(prefixes: string[], r: number, loc: number[], shape: number[], nda: ndarray, excludes: [number, number][]) {
    if (r != shape.length-1) {
        console.log(`${prefixes[r]}[`);

        for (loc[r] = 0; loc[r] < shape[r]; ) {
            if (excludes && loc[r] == excludes[r][0]) {
                console.log(`${prefixes[r+1]}...`);
                loc[r] = (excludes[r][1] > 0) ? (excludes[r][1] - 1) : (shape[r] - 1 + excludes[r][1]); 
            }
            else {
                printNdarrayRecursive(prefixes, r+1, loc, shape, nda, excludes);
            }
            ++loc[r];
        }
 
        const tailComma = (r > 0 && loc[r-1] != shape[r-1] - 1) ? "," : "";
        console.log(`${prefixes[r]}]${tailComma}`);
    }
    else {
        let line = prefixes[r] + '[';
        for (loc[r] = 0; loc[r] < shape[r]; ) {
            if (excludes && loc[r] == excludes[r][0]) {
                line += ' ... , ';
                loc[r] = (excludes[r][1] > 0) ? (excludes[r][1] - 1) : (shape[r] - 1 + excludes[r][1]); 
            }
            else {
                const v = nda.get(...loc);
                line += v;
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

export function printNdarray(nda: ndarray, excludes: [number, number][], name: string = '') {
    const shape = nda.shape;
    const rank = shape.length;
    const loc = new Array(rank).fill(0);
    const spacePrefix = new Array(rank).fill("");
    for (let i = 1; i < rank; ++i) {
        spacePrefix[i] = `${spacePrefix[i-1]}  `;
    }
    
    console.log(`${name} of shape:${shape} = `);
    printNdarrayRecursive(spacePrefix, 0, loc, shape, nda, excludes);
}
