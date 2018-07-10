"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function isTypedArray(a) {
    return a instanceof Float32Array || a instanceof Int32Array || a instanceof Uint8Array;
}
exports.isTypedArray = isTypedArray;
function getShape(val) {
    if (isTypedArray(val)) {
        return [val.length];
    }
    if (!Array.isArray(val)) {
        return [];
    }
    const shape = [];
    while (val instanceof Array) {
        shape.push(val.length);
        val = val[0];
    }
    return shape;
}
exports.getShape = getShape;
const ArrayDataInstances = {
    float32: new Float32Array([1.0]),
    int32: new Int32Array([1]),
    bool: new Uint8Array([0]),
};
function getTypedArraySample(dtype) {
    return ArrayDataInstances[dtype];
}
exports.getTypedArraySample = getTypedArraySample;
function toTypedArray(a, dtype) {
    if ((a instanceof Float32Array && dtype === 'float32') ||
        (a instanceof Int32Array && dtype === 'int32') ||
        (a instanceof Uint8Array && dtype === 'bool')) {
        return a;
    }
    if (Array.isArray(a)) {
        const arr = flatten(a);
        if (dtype === 'float32' || dtype === 'int32') {
            return new Float32Array(arr);
        }
        else if (dtype === 'bool') {
            const bools = new Uint8Array(arr.length);
            for (let i = 0; i < bools.length; ++i) {
                bools[i] = (Math.round(arr[i]) !== 0) ? 1 : 0;
            }
            return bools;
        }
    }
    throw new Error('should not arrived here!');
}
exports.toTypedArray = toTypedArray;
function flatten(arr, ret = []) {
    if (Array.isArray(arr)) {
        for (let i = 0; i < arr.length; ++i) {
            flatten(arr[i], ret);
        }
    }
    else {
        ret.push(arr);
    }
    return ret;
}
exports.flatten = flatten;
//# sourceMappingURL=types.js.map