"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var TensorEngine = (function () {
    function TensorEngine(backend) {
        this.backend = backend;
    }
    TensorEngine.prototype.register = function (t) {
        throw new Error("Method not implemented.");
    };
    TensorEngine.prototype.dispose = function (t) {
        throw new Error("Method not implemented.");
    };
    return TensorEngine;
}());
exports.TensorEngine = TensorEngine;
//# sourceMappingURL=engine.js.map