import { assert } from 'chai';

import * as tf from '../src/index';
import * as ndarray from 'ndarray';
import * as nda_gemm from 'ndarray-gemm';

describe("_NdArrayMatMul", function() {

    for (let w = 32; w <= 1024; w*=2) {
        const startInitA = new Date().getTime();
        const a = tf.randomNorm([w, w], 2.0, 3.0, 'float32', 10000);
        const nda = ndarray(tf.readSync(a), [w, w]);
        const msInitA = (new Date()).getTime() - startInitA;
        console.log(`  Finish initialize A[${w}*${w}]  in ${msInitA}ms.`);
    
        const startInitB = new Date().getTime();
        const b = tf.randomNorm([w, w], 1.0, 2.0, 'float32', 20000);
        const ndb = ndarray(tf.readSync(b), [w, w]);
        const msInitB = (new Date()).getTime() - startInitB;
        console.log(`  Finish initialize B[${w}*${w}] in ${msInitB}ms.`);
        
        const startInitC = new Date().getTime();
        const C = new Float32Array(w*w);
        const ndc = ndarray(C, [w,w]);
        const msInitC = (new Date()).getTime() - startInitC;
        console.log(`  Finish allocat C[${w}*${w}] in ${msInitC}ms.`);

        console.log('start mat mul............');
        const startMatMul = new Date().getTime();
        nda_gemm(ndc, nda, ndb);
        let msMatMul = (new Date()).getTime() - startMatMul;
        console.log(`  Finish MatMul in ${msMatMul}ms. result is of shape: ${w}x${w}`);

        it(`a[${w}*${w}] * b[${w}*${w}]`, function() {
            assert(true, "");
        }).timeout(200000);
    }
});