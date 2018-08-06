import * as tf from '../../src/index';

import * as ndarray from 'ndarray';
import * as nda_gemm from 'ndarray-gemm';

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
    C.fill(0);
    const ndc = ndarray(C, [w,w]);
    const msInitC = (new Date()).getTime() - startInitC;
    console.log(`  Finish allocat C[${w}*${w}] in ${msInitC}ms.`);

    console.log('start mat mul............');
    const startMatMul = new Date().getTime();
    nda_gemm(ndc, nda, ndb);
    let msMatMul = (new Date()).getTime() - startMatMul;
    console.log(`  Finish MatMul in ${msMatMul}ms. result is of shape: ${w}x${w}`);

}

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
}    



import { NDView } from "../../src/NdView/ndview";

let a = new NDView([1, 2, 3, 2, 2, 3], [1, 2, 3]);
a.print('a');

let atrans = a.transpose([2, 1, 0]);
atrans.print('a.transpose([2,1,0])');

let apick = a.pick([-1, -1, 1]);
apick.print('a.pick([-1,-1,1])');

let aexp = a.unsqueeze([2,3,3]);
aexp.print('a.unsqueeze([2,3,3])');

let asqueeze = a.squeeze();
asqueeze.print('a.squeeze()');

const paddings = [[1, 1], [1, 1]];
let apad = asqueeze.pad(paddings as [number, number][]);
apad.print(`asqueeze.pad(${JSON.stringify(paddings)})`);

let areshape = a.reshape([-1, 2]);
areshape.print('a.reshape([-1,2])');

let aflat = areshape.reshape([-1]);
aflat.print('areshape.reshape([-1])');

let atile = asqueeze.tile([1,2]);
atile.print('asqueeze.title([0,2])');

//import * as tf from '../../src/index';

let ta = tf.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
let tb = tf.tensor([1, 1, 1], [3, 1]);
let tc = tf.matMul(ta, tb, false, false);
tc.print();


// let s = new NDView(["good", "good", "study", "day", "day", "up"], [1, 2, 3]);
// s.print('s');

// let strans = s.transpose([2, 1, 0]);
// strans.print('s-transpose');

// let spick = s.pick([-1, -1, 1]);
// spick.print('s-pick');

