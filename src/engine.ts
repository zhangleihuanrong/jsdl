import { Tensor } from './tensor';
import { Backend } from './backend';
import { TensorManager } from './tensor_manager';
import { DataType, Shape, StrictTensorLike } from './types';

export class TensorEngine implements TensorManager {
    backend : Backend;

    constructor(backend: Backend) {
        this.backend = backend;
    }

    wrap(t: Tensor, dtype: DataType, shape: Shape, backendTensor: object): void {
        // engine's logic without touching backend
        this.backend.wrap(t, dtype, shape, backendTensor);
    }

    make(t: Tensor, dtype: DataType, shape: Shape, values: StrictTensorLike): void {
        // TODO: engine's logic without touching backend
        this.backend.make(t, dtype, shape, values);
    }

    free(t: Tensor): void {
        this.backend.free(t);
    }
}
