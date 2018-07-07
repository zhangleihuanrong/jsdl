import { TypedArray, BackendTensor } from './types';
import { Tensor } from './tensor';
import { Backend } from './backend';
import { TensorManager } from './tensor_manager';

export type ForwardFunc = (backend: Backend, save? : (t: Tensor) => Tensor) => Tensor;

export class TensorEngine implements TensorManager {
    backend : Backend;

    constructor(backend: Backend) {
        this.backend = backend;
    }

    register(t: Tensor) {

    }

    dispose(t: Tensor) {

    }

}
