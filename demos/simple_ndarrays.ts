import * as ndarray from 'ndarray';
import * as nd_ops from 'ndarray-ops';
import { printNdarray } from '../src/utils/ndarray_print';

const a = ndarray(new Float32Array([1, 2, 3, 2, 2, 3]), [2, 3]);
a.set(0, 0, 10000);
a.set(1, 2, 20000);
printNdarray(a, 'a');

const b = ndarray(new Float32Array(6), [2, 3]);
printNdarray(b, 'b');
nd_ops.assign(b, a);
printNdarray(b, 'b after assign by a');

const fa = new Float32Array(50);
const shape = [ 5, 10 ];
const c = ndarray(fa, shape);
printNdarray(c, 'c');

