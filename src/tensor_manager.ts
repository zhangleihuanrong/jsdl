import { DataType, StrictTensorLike, Shape, BackendTensor } from './types';
import { Tensor } from './tensor';

export interface TensorManager {
    wrap(t: Tensor, dtype: DataType, shape: Shape, backendTensor: BackendTensor) : void;
    make(t: Tensor, dtype: DataType, shape: Shape, values: StrictTensorLike) : void;
    free(t: Tensor): void;
}
