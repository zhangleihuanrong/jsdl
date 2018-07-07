"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var engine_1 = require("./engine");
var Environment = (function () {
    function Environment() {
        this.backendRegistry = {};
    }
    Environment.prototype.registerBackend = function (name, backend, score) {
        if (score === void 0) { score = 1; }
        if (name in this.backendRegistry) {
            console.warn("Tensor backend '" + name + "' already registered.");
        }
        else {
            this.backendRegistry[name] = { backend: backend, score: score };
        }
    };
    Environment.prototype.findBackend = function (name) {
        if (!(name in this.backendRegistry)) {
            return null;
        }
        return this.backendRegistry[name].backend;
    };
    Environment.prototype.getBestBackend = function () {
        var bestName;
        var highestScore = -1.0;
        for (var name_1 in this.backendRegistry) {
            if (this.backendRegistry[name_1].score > highestScore) {
                bestName = name_1;
                highestScore = this.backendRegistry[name_1].score;
            }
        }
        return bestName;
    };
    Environment.prototype.useBackend = function (name) {
        this.currentBackendName = name;
        var backend = this.findBackend(name);
        this.currentEngine = new engine_1.TensorEngine(backend);
    };
    Object.defineProperty(Environment.prototype, "engine", {
        get: function () {
            if (this.currentEngine == null) {
                this.useBackend(this.getBestBackend());
            }
            return this.currentEngine;
        },
        enumerable: true,
        configurable: true
    });
    return Environment;
}());
exports.Environment = Environment;
function getGlobalNamespace() {
    var ns;
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
    var ns = getGlobalNamespace();
    ns.__global_TensorEnv = ns.__global_TensorEnv || new Environment();
    return ns.__global_TensorEnv;
}
exports.ENV = getOrCreateEnvironment();
exports.engine = exports.ENV.engine;
//# sourceMappingURL=environments.js.map