"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class TensorEngine {
    register(t) {
        this.backend.register(t);
    }
    dispose(t) {
        this.backend.dispose(t);
    }
    constructor(backend) {
        this.backend = backend;
    }
}
exports.TensorEngine = TensorEngine;
//# sourceMappingURL=engine.js.map