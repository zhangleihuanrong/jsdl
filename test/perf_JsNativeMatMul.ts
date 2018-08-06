import { assert } from 'chai';

import * as tf from '../src/index';

describe("NativeJsMatMul", function() {

    for (let w = 32; w <= 1024; w*=2) {
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

        console.log('start mat mul............');
        const startMatMul = new Date().getTime();
        const A = a.data;
        const B = b.data;
        for (let blk0 = w-32; blk0 >= 0; blk0 -= 32) {
            for (let blk1 = w-32; blk1 >= 0; blk1 -= 32) {
                for (let blk2 = w-32; blk2 >= 0; blk2 -= 32) {
                    let oc = blk0 * w + blk1;
                    for (let i = 0; i < 32; ++i) {
                        for (let j = 0; j < 32; ++j) {
                            let aa = (blk0 + i) * w + blk2;
                            let bb = blk2 * w + (blk1 + j);
                            let sum = 0.0;
                            for (let k = 0; k < 32; ++k) {
                                sum += A[aa] * B[bb];
                                aa += 1;
                                bb += w;
                            }
                            C[oc] += sum;
                            oc += 1;
                        }
                        oc += w;
                    }
                }
            }
        }

        let msMatMul = (new Date()).getTime() - startMatMul;
        console.log(`  Finish MatMul in ${msMatMul}ms. result is of shape: ${w}x${w}`);
        
        it(`a[${w}*${w}] * b[${w}*${w}]`, function() {
            assert(true, "");
        }).timeout(200000);
    }
});