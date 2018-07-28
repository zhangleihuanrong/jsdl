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
