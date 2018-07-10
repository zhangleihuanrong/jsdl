"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const ndarray = require("ndarray");
function ndarray_print2(arr) {
    console.log(`  ==shape:${arr.shape}`);
    for (let y = 0; y < arr.shape[0]; ++y) {
        let line = "";
        for (let x = 0; x < arr.shape[1]; ++x) {
            line += " " + arr.get(y, x);
        }
        console.log("...   ", line);
    }
}
const a = ndarray([1, 2, 3, 2, 2, 3], [2, 3]);
ndarray_print2(a);
const b = a.transpose(1, 0);
ndarray_print2(b);
a.set(0, 0, 10000);
ndarray_print2(a);
ndarray_print2(b);
//# sourceMappingURL=sand.js.map