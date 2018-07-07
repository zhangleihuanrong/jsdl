"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environments_1 = require("../environments");
var BackendJsCpu = (function () {
    function BackendJsCpu() {
    }
    BackendJsCpu.prototype.write = function (bt, values) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.read = function (bt) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.readSync = function (bt) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.matMul = function (a, b, transposeA, transposeB) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.transpose = function (x, perm) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.add = function (a, b) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.neg = function (a) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.multiply = function (a, b) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.relu = function (x) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.register = function (t) {
        throw new Error("Method not implemented.");
    };
    BackendJsCpu.prototype.dispose = function (t) {
        throw new Error("Method not implemented.");
    };
    return BackendJsCpu;
}());
var backendJsCpu = new BackendJsCpu();
environments_1.ENV.registerBackend('JS_CPU', backendJsCpu, 1);
//# sourceMappingURL=backend_js.js.map