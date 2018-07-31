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

// let s = new NDView(["good", "good", "study", "day", "day", "up"], [1, 2, 3]);
// s.print('s');

// let strans = s.transpose([2, 1, 0]);
// strans.print('s-transpose');

// let spick = s.pick([-1, -1, 1]);
// spick.print('s-pick');

