import {assert} from 'chai';
import * as ndarray from 'ndarray';
import * as nd_ops from 'ndarray-ops';
import { printNdarray } from '../src/utils/ndarray_print';


describe('Simple Test in JS', function() {
    it('Simple ndarray program should work!', function() {
        let a = ndarray(new Float32Array([1, 2, 3, 2, 2, 3]), [1, 2, 3]);
        printNdarray(a, 'a');
        
        function rangedArray(s: number) : number[] {
            const arr = new Array(s);
            for (let i = 0; i < s; ++i) arr[i] = i;
            return arr;
        }
        
        function broadCastedNdarray(a: ndarray, newShape: number[]) : ndarray {
            a = a.transpose(a, rangedArray(a.shape.length));
            const shape : number[] = a.shape;
            const strides : number[] = a.stride;
            shape.forEach((orig, i) => {
                if (orig == 1 && newShape[i] > 1) {
                    strides[i] = 0;
                }
            });
            a.shape = newShape;
            return a;
        }
        
        a = broadCastedNdarray(a, [2, 2, 3]);
        printNdarray(a, 'a-broadcasted');
        
        a = a.transpose(2, 1, 0);
        printNdarray(a, 'a-transpose');
        
        a = ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [1, 2, 3]);
        
        a.set(0, 0, 0, 10000);
        a.set(0, 1, 2, 20000);
        printNdarray(a, 'a');
        a = a.pick(0, null, null);
        
        const b = ndarray(new Float32Array(6), [2, 3]);
        printNdarray(b, 'b');
        nd_ops.assign(b, a);
        printNdarray(b, 'b after assign by a');
        
        const fa = new Float32Array(50);
        const shape = [ 5, 10 ];
        const c = ndarray(fa, shape);
        printNdarray(c, 'c');

        assert.equal('Good Luck!', 'Good' + ' Luck!');
    });
});
