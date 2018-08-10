export function assert(expr: boolean, msg: string | (() => string)) {
    if (!expr) {
        throw new Error(typeof msg === 'string' ? msg : msg());
    }
}

export function isScalarShape(shape: number[]): boolean {
    return (shape.length === 0);
}

export type FlatArrayLike = number[] | boolean[] | string[] | Float32Array | Int32Array | Uint8Array | Uint8ClampedArray;

export function areArraysEqual<T extends FlatArrayLike>(n1: T, n2: T) {
    if (n1.length !== n2.length) {
        return false;
    }
    for (let i = 0; i < n1.length; i++) {
        if (n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}


export function simpleHash32(s: string) : number {
    let hash = 0;
    if (s.length === 0) return hash;
    for (let i = 0; i < s.length; i++) {
        const chr = s.charCodeAt(i);
        hash  = ((hash << 5) - hash) + chr;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
}


export function shuffle(array: any[]|Uint32Array|Int32Array|Float32Array|Uint8Array): void {
    let counter = array.length;
    let temp = 0;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = (Math.random() * counter) | 0;
        // Decrease counter by 1
        counter--;
        // And swap the last element with it
        temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }
}
