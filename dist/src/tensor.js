"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const types_1 = require("./types");
const environments_1 = require("./environments");
class Tensor {
    constructor(shape, dtype, values = null, backendTensor = null) {
        this.id = Tensor.sNextId++;
        this.shape = shape;
        this.dtype = dtype;
        this.size = (shape && shape.length > 0) ? shape.reduce((a, b) => a * b, 1) : 0;
        this.backendTensor = backendTensor;
        environments_1.ENV.engine.register(this);
        environments_1.ENV.engine.backend.write(this, values);
    }
    static create(values, shape = null, dtype = 'float32') {
        const sa = (!types_1.isTypedArray(values) && !Array.isArray(values)) ?
            [values] :
            values;
        shape = shape || types_1.getShape(sa);
        return new Tensor(shape, dtype, sa);
    }
    get rank() {
        return this.shape.length;
    }
}
Tensor.sNextId = 0;
exports.Tensor = Tensor;
//# sourceMappingURL=tensor.js.map