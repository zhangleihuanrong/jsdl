"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var types_1 = require("./types");
var environments_1 = require("./environments");
var ndarray = require("ndarray");
var Tensor = (function () {
    function Tensor(shape, dtype, values, backendTensor) {
        if (values === void 0) { values = null; }
        if (backendTensor === void 0) { backendTensor = null; }
        this.id = Tensor.sNextId++;
        this.shape = shape;
        this.dtype = dtype;
        if (values != null) {
            this.tensor = ndarray(values, shape);
        }
        else {
            this.tensor = null;
        }
        this.backendTensor = backendTensor;
        environments_1.ENV.engine.register(this);
        environments_1.ENV.engine.backend.write(backendTensor, values);
    }
    Object.defineProperty(Tensor.prototype, "rank", {
        get: function () {
            return this.shape.length;
        },
        enumerable: true,
        configurable: true
    });
    Tensor.create = function (values, shape, dtype) {
        if (shape === void 0) { shape = null; }
        if (dtype === void 0) { dtype = 'float32'; }
        var fa = (!types_1.isTypedArray(values) && !Array.isArray(values)) ?
            [values] : values;
        shape = shape || types_1.getShape(fa);
        var ta = types_1.toTypedArray(fa, dtype);
        return new Tensor(shape, dtype, ta);
    };
    Tensor.sNextId = 0;
    return Tensor;
}());
exports.Tensor = Tensor;
//# sourceMappingURL=tensor.js.map