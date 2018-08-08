import * as tf from '../../src';

import * as ndarray from 'ndarray';
import * as nda_gemm from 'ndarray-gemm';

export function squareMatMul(C: Float32Array, A: Float32Array, B: Float32Array, w: number) {
    C.fill(0);
    const _b = 32;
    for (let z0 = w - _b; z0 >= 0; z0 -= _b) {
        for (let z1 = w - _b; z1 >= 0; z1 -= _b) {
            let cob = z0 * w + z1;
            for (let z2 = w - _b; z2 >= 0; z2 -= _b) {
                let bob = z2 * w + z1;
                let aob = z0 * w + z2;

                for (let i = 0; i < _b; ++i) {
                    let co = cob + i * w;
                    let ao = aob + i * w;
                    for (let j = 0; j < _b; ++j) {
                        let bo = bob + j;
                        let r = 0.0;
                        // C[(z0+i)*w + (z1 + j)] = sum(A[(z0+i)*w + (z2 + k)] * B[(z2 + k) * w + (z1+j)])
                        for (let k = 0; k < _b; ++k) {
                            r += A[ao+k] * B[bo];
                            bo += w;
                        }
                        C[co++] += r;
                    }
                }
            }
        }
    }
}

export function squareMatMulNoIndexCal(C: Float32Array, A: Float32Array, B: Float32Array, w: number) {
    let co = 0;
    let arow = 0;
    for (let i = 0; i < w; ++i) {
        for (let j = 0; j < w; ++j) {
            let ao = arow;
            let bo = j;
            let r = 0;
            for (let k = 0; k < w; ++k) {
                r += A[ao++] * B[bo];
                bo += w;
            }
            C[co++] = r;
        }
        arow += w;
    }
}

export function squareMatMulRaw(C: Float32Array, A: Float32Array, B: Float32Array, w: number) {
    for (let i = 0; i < w; ++i) {
        for (let j = 0; j < w; ++j) {
            let r = 0;
            for (let k = 0; k < w; ++k) {
                r += A[i * w + k] * B[k * w + j];
            }
            C[i * w + j] = r;
        }
    }
}

const sampleAs = new Map<number, Float32Array>();
const sampleBs = new Map<number, Float32Array>();

for (let w = 32; w <= 2048; w*=2) {
    const startInitA = new Date().getTime();
    const a = tf.randomNorm([w, w], 2.0, 3.0, 'float32', 10000);
    sampleAs[w] = tf.read(a);
    const msInitA = (new Date()).getTime() - startInitA;
    console.log(`  Finish initialize A[${w}*${w}]  in ${msInitA}ms.`);

    const startInitB = new Date().getTime();
    const b = tf.randomNorm([w, w], 1.0, 2.0, 'float32', 20000);
    sampleBs[w] = tf.read(b);
    const msInitB = (new Date()).getTime() - startInitB;
    console.log(`  Finish initialize B[${w}*${w}] in ${msInitB}ms.`);
}

console.log("===========================================================");
console.log("================SQUARE MATRIX MATMUL=======================");
console.log("===========================================================");
for (let w = 32; w <= 2048; w*=2) {
    const C = new Float32Array(w*w);
    C.fill(0);
    const ndc = ndarray(C, [w,w]);
    const D = new Float32Array(w*w);
    D.fill(0);

    console.log(`${w}x${w}......`);

    const nda = ndarray(sampleAs[w], [w, w]);
    const ndb = ndarray(sampleBs[w], [w, w]);
    const startMatMul = new Date().getTime();
    nda_gemm(ndc, nda, ndb);
    let msMatMul = (new Date()).getTime() - startMatMul;
    console.log(`    ${w}x${w} ndarrray matMul: ${msMatMul}ms.  `);

    const startMatMulNative = new Date().getTime();
    squareMatMul(D, sampleAs[w], sampleBs[w], w);
    let msMatMulNative = (new Date()).getTime() - startMatMulNative;
    console.log(`    ${w}x${w} js native matMul: ${msMatMulNative}ms.  `);

    let diffCount = 0;
    for (let i = 0, sz = w*w; i < sz && diffCount < 10; ++i) {
        const diff = Math.abs(C[i] - D[i]);
        const bigger = Math.max(Math.abs(C[i]), Math.abs(D[i]));
        if (diff > 0.001 && (diff / (bigger + diff)) > 0.001) {
            console.warn(`NdarrayResult[${i}] = ${C[i]}  vs JSNativeResult[${i}] = ${D[i]}`);
            ++diffCount;
        }
    }
}


// import { NDView } from "../../src/NdView/ndview";

// let a = new NDView([1, 2, 3, 2, 2, 3], [1, 2, 3]);
// a.print('a');

// let atrans = a.transpose([2, 1, 0]);
// atrans.print('a.transpose([2,1,0])');

// let apick = a.pick([-1, -1, 1]);
// apick.print('a.pick([-1,-1,1])');

// let aexp = a.unsqueeze([2,3,3]);
// aexp.print('a.unsqueeze([2,3,3])');

// let asqueeze = a.squeeze();
// asqueeze.print('a.squeeze()');

// const paddings = [[1, 1], [1, 1]];
// let apad = asqueeze.pad(paddings as [number, number][]);
// apad.print(`asqueeze.pad(${JSON.stringify(paddings)})`);

// let areshape = a.reshape([-1, 2]);
// areshape.print('a.reshape([-1,2])');

// let aflat = areshape.reshape([-1]);
// aflat.print('areshape.reshape([-1])');

// let atile = asqueeze.tile([1,2]);
// atile.print('asqueeze.title([0,2])');

// //import * as tf from '../../src/index';

// let ta = tf.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
// let tb = tf.tensor([1, 1, 1], [3, 1]);
// let tc = tf.matMul(ta, tb, false, false);
// tc.print();


// // let s = new NDView(["good", "good", "study", "day", "day", "up"], [1, 2, 3]);
// // s.print('s');

// // let strans = s.transpose([2, 1, 0]);
// // strans.print('s-transpose');

// // let spick = s.pick([-1, -1, 1]);
// // spick.print('s-pick');

