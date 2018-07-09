//import { TypedArray, BackendTensor } from './types';
import { Tensor } from './tensor';
import { Backend } from './backend';
import { TensorManager } from './tensor_manager';

export type ForwardFunc = (backend: Backend, save? : (t: Tensor) => Tensor) => Tensor;

export class TensorEngine implements TensorManager {
    register(t: Tensor): void {
        this.backend.register(t);
    }
    dispose(t: Tensor): void {
        this.backend.dispose(t);
    }
    backend : Backend;

    constructor(backend: Backend) {
        this.backend = backend;
    }
}
