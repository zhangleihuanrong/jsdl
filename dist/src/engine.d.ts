import { Tensor } from './tensor';
import { Backend } from './backend';
import { TensorManager } from './tensor_manager';
export declare type ForwardFunc = (backend: Backend, save?: (t: Tensor) => Tensor) => Tensor;
export declare class TensorEngine implements TensorManager {
    register(t: Tensor): void;
    dispose(t: Tensor): void;
    backend: Backend;
    constructor(backend: Backend);
}
