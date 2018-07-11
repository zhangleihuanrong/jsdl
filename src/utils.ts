import { Tensor } from './tensor';
import * as tf from './ops';

import * as ndarray from 'ndarray';

function ndarray_print2d(arr, name: string = '', heading: number = 9, tailing: number = 6) {
    console.log(`  ==shape for array ${name} is ${arr.shape}`)
    for (let y = 0; y < arr.shape[0]; ) {
        let line = "";
        if (y >= heading && y < arr.shape[0] - tailing) {
            line = " ... ";
            y = arr.shape[0] - tailing;
        }
        else {
            for (let x = 0; x < arr.shape[1]; ++x) {
                line += " " + arr.get(y, x) ;
            }
            ++y;
        }
        console.log("   ", line);
    }
}

export function print(t: Tensor, heading: number = 5, tailing: number = 3) {
    const ta = tf.readSync(t);
    const arr = ndarray(ta, t.shape);

    ndarray_print2d(arr);
}
