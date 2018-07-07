import {Tensor} from './tensor';

export interface TensorManager {
    register(t: Tensor) : void;
    dispose(t: Tensor): void;
}
