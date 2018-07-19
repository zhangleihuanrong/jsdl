import { Tensor } from './tensor';
import { Backend } from './backend';
import { TensorManager } from './tensor_manager';
import { DataType, Shape, StrictTensorLike, BackendTensor } from './types';

export class TensorEngine implements TensorManager {
    backend : Backend;

    constructor(backend: Backend) {
        this.backend = backend;
    }

    wrap(t: Tensor, backendTensor: BackendTensor): void {
        this.backend.wrap(t, backendTensor);
        // TODO: Other's engine's logic without touching backend
    }

    make(t: Tensor, dtype: DataType, shape: Shape, values: StrictTensorLike): void {
        this.backend.make(t, dtype, shape, values);
        // TODO: engine's logic without touching backend
    }

    free(t: Tensor): void {
        this.backend.free(t);
    }
}
