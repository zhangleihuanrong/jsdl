import { assert } from 'chai';

import * as tf from '../src';

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

describe("NativeJsMatMul", function() {
    for (let w = 32; w <= 2048; w*=2) {
        const startInitA = new Date().getTime();
        let a = tf.randomNorm([w, w], 2.0, 3.0, 'float32', 10000);
        const msInitA = (new Date()).getTime() - startInitA;
        console.log(`  Finish initialize A[${w}*${w}]  in ${msInitA}ms.`);
    
        const startInitB = new Date().getTime();
        let b = tf.randomNorm([w, w], 1.0, 2.0, 'float32', 20000);
        let msInitB = (new Date()).getTime() - startInitB;
        console.log(`  Finish initialize B[${w}*${w}] in ${msInitB}ms.`);
        
        const startInitC = new Date().getTime();
        const C = new Float32Array(w*w);
        C.fill(0);
        let msInitC = (new Date()).getTime() - startInitC;
        console.log(`  Finish allocat C[${w}*${w}] in ${msInitC}ms.`);

        console.log('  Start mat mul............');
        const startMatMul = new Date().getTime();
        const A = tf.read(a);
        const B = tf.read(b);
        squareMatMul(C, A, B, w);
        let msMatMul = (new Date()).getTime() - startMatMul;
        console.log(`  Finish MatMul in ${msMatMul}ms. result is of shape: ${w}x${w}`);
        
        it(`a[${w}*${w}] * b[${w}*${w}]`, function() {
            assert(true, "");
        }).timeout(200000);
    }
});
