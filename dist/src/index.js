"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tensor_1 = require("./tensor");
const environments_1 = require("./environments");
const transpose = environments_1.ENV.engine.backend.transpose;
exports.transpose = transpose;
const matMul = environments_1.ENV.engine.backend.matMul;
exports.matMul = matMul;
const neg = environments_1.ENV.engine.backend.neg;
exports.neg = neg;
const add = environments_1.ENV.engine.backend.add;
exports.add = add;
const multiply = environments_1.ENV.engine.backend.multiply;
exports.multiply = multiply;
const relu = environments_1.ENV.engine.backend.relu;
exports.relu = relu;
const tensor = tensor_1.Tensor.create;
exports.tensor = tensor;
//# sourceMappingURL=index.js.map