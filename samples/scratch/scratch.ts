import { NDView } from "../../src/NdView/ndview";



let a = new NDView(new Float32Array([1, 2, 3, 2, 2, 3]), [1, 2, 3]);
a.print();


let atrans = a.transpose([2, 1, 0]);
atrans.print('a-transpose');

let apick = a.pick([-1, -1, 1]);
apick.print('a-pick');


