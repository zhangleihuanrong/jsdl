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

aexp = a.unsqueeze([]);
aexp.print('a.unsqueeze([])');


// let s = new NDView(["good", "good", "study", "day", "day", "up"], [1, 2, 3]);
// s.print('s');

// let strans = s.transpose([2, 1, 0]);
// strans.print('s-transpose');

// let spick = s.pick([-1, -1, 1]);
// spick.print('s-pick');

