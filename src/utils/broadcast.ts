import {assert as ASSERT } from './gadget';

export function bidirectionBroadcast(s1: number[], s2: number[]) : number[] {
    if (s1.length != s2.length) {
        const dd = Math.abs(s2.length - s1.length);
        if (s1.length < s2.length) {
            s1 = s2.map((v, i) => (i < dd) ? 1 : s1[i - dd]);
        }
        else {
            s2 = s1.map((v, i) => (i < dd) ? 1 : s2[i - dd]);
        }
    }
    const castable = s2.every((v, i) => (v == s1[i] || v == 1 || s1[i] == 1));

    if (!castable) return null;
    return s2.map((v, i) => (v >= s1[i])? v : s1[i]);
}

export function canBroadcastTo(bcTo: number[], bcFrom: number[]) {
    if (bcFrom.length < bcTo.length) {
        const dd = bcTo.length - bcFrom.length;
        bcFrom = bcTo.map((v, i) => (i < dd)? 1 : bcFrom[i-dd]);
    }
    return (bcFrom.length == bcTo.length && bcTo.every((v, i) => (v == bcFrom[i] || bcFrom[i] == 1)));
}

export function getUnsqueezeAxisForBroadcast(bcTo: number[], bcFrom: number[]) : number[] {
    if (bcFrom.length < bcTo.length) {
        return new Array(bcTo.length - bcFrom.length).fill(0);
    }
    return null;
}

export function getUnsqueezedShapeForBroadcast(bcTo: number[], bcFrom: number[]) : number[] {
    if (bcFrom.length < bcTo.length) {
        return (new Array(bcTo.length - bcFrom.length).fill(1)).concat(...bcFrom);
    }
    return bcFrom;
}

export function getBroadcastRepeats(bcTo: number[],  bcFrom: number[]) : number[] {
    ASSERT(bcFrom.length == bcTo.length && bcTo.every((v, i) => (v == bcFrom[i] || bcFrom[i] == 1)), 
            "Can not broadcast to!");
    const repeats = bcTo.map((v, i) => (v > bcFrom[i]) ? v : 1);
    return (repeats.every(v => v == 1)) ? null : repeats;
}

export function areShapesEqual(shapeA: number[], shapeB: number[]) : boolean {
    return (shapeA.length == shapeB.length && shapeA.every((v, i) => v === shapeB[i]));
}
