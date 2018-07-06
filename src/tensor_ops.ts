import { TensorLike, DataType } from "./types";
import { Tensor } from './tensor';

export class TensorOps {
    static tensor(values: TensorLike, shape: number[], dtype: DataType = 'float32') : Tensor {
        // TBD
        return new Tensor(shape, dtype, values)
    }
}
