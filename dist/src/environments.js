"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const engine_1 = require("./engine");
class Environment {
    constructor() {
        this.backendRegistry = {};
    }
    registerBackend(name, backend, score = 1) {
        if (name in this.backendRegistry) {
            console.warn(`Tensor backend '${name}' already registered.`);
        }
        else {
            this.backendRegistry[name] = { backend, score };
        }
    }
    findBackend(name) {
        if (!(name in this.backendRegistry)) {
            return null;
        }
        return this.backendRegistry[name].backend;
    }
    getBestBackend() {
        let bestName;
        let highestScore = -1.0;
        for (let name in this.backendRegistry) {
            if (this.backendRegistry[name].score > highestScore) {
                bestName = name;
                highestScore = this.backendRegistry[name].score;
            }
        }
        return bestName;
    }
    useBackend(name) {
        this.currentBackendName = name;
        const backend = this.findBackend(name);
        this.currentEngine = new engine_1.TensorEngine(backend);
    }
    get engine() {
        if (this.currentEngine == null) {
            this.useBackend(this.getBestBackend());
        }
        return this.currentEngine;
    }
}
exports.Environment = Environment;
function getGlobalNamespace() {
    let ns;
    if (typeof (window) !== 'undefined') {
        ns = window;
    }
    else if (typeof (global) !== 'undefined') {
        ns = global;
    }
    else {
        throw new Error('Could not find global namespace!');
    }
    return ns;
}
function getOrCreateEnvironment() {
    const ns = getGlobalNamespace();
    ns.__global_TensorEnv = ns.__global_TensorEnv || new Environment();
    return ns.__global_TensorEnv;
}
exports.ENV = getOrCreateEnvironment();
exports.engine = exports.ENV.engine;
//# sourceMappingURL=environments.js.map