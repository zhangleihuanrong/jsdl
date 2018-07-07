
import { Backend } from '../backend';
import { ENV } from '../environments';

//import ndarray from 'ndarray';
import { Tensor } from '../tensor';

class BackendJsCpu implements Backend {
    write(bt: object, values: Float32Array | Int32Array | Uint8Array): void {
        throw new Error("Method not implemented.");
    }
    read(bt: object): Promise<Float32Array | Int32Array | Uint8Array> {
        throw new Error("Method not implemented.");
    }
    readSync(bt: object): Float32Array | Int32Array | Uint8Array {
        throw new Error("Method not implemented.");
    }
    matMul(a: Tensor, b: Tensor, transposeA: boolean, transposeB: boolean): Tensor {
        throw new Error("Method not implemented.");
    }
    transpose(x: Tensor, perm: number[]): Tensor {
        throw new Error("Method not implemented.");
    }
    add(a: Tensor, b: Tensor): Tensor {
        throw new Error("Method not implemented.");
    }
    neg(a: Tensor): Tensor {
        throw new Error("Method not implemented.");
    }
    multiply(a: Tensor, b: Tensor): Tensor {
        throw new Error("Method not implemented.");
    }
    relu(x: Tensor): Tensor {
        throw new Error("Method not implemented.");
    }
    register(t: Tensor): void {
        throw new Error("Method not implemented.");
    }
    dispose(t: Tensor): void {
        throw new Error("Method not implemented.");
    }
}

const backendJsCpu = new BackendJsCpu() as Backend;

ENV.registerBackend('JS_CPU', backendJsCpu, 1);
