"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const types_1 = require("../types");
const tensor_1 = require("../tensor");
const environments_1 = require("../environments");
const ndarray_1 = require("ndarray");
const ndarray_gemm_1 = require("ndarray-gemm");
const ndarray_ops_1 = require("ndarray-ops");
class NdarrayTensor {
    constructor(arr) {
        this.arr = arr;
    }
}
;
class JsNdarrayBackend {
    register(t) {
        if (t.backendTensor == null) {
            const bt = new NdarrayTensor(null);
            t.backendTensor = bt;
        }
    }
    dispose(t) {
        if (t.backendTensor) {
            const bt = t.backendTensor;
            delete bt.arr;
        }
    }
    write(t, values) {
        const bt = t.backendTensor;
        const taSample = types_1.getTypedArraySample(t.dtype);
        if (!values) {
            const ta = (t.dtype == 'float32') ? new Float32Array(t.size) :
                (t.dtype == 'int32') ? new Int32Array(t.size) :
                    new Uint8Array(t.size);
            bt.arr = ndarray_1.default(ta, t.shape);
        }
        else if (values instanceof Array) {
            bt.arr = ndarray_1.default(values, t.shape);
        }
        else if (values.constructor == taSample.constructor) {
            bt.arr = ndarray_1.default(values, t.shape);
        }
        else {
            throw new Error("Method not implemented.");
        }
    }
    read(t) {
        return new Promise((resolve, reject) => {
            resolve(this.readSync(t));
        });
    }
    readSync(t) {
        if (t.backendTensor) {
            const bt = t.backendTensor;
            return types_1.toTypedArray(bt.arr.data, t.dtype);
        }
        return null;
    }
    transpose(x, perm) {
        const bt = x.backendTensor;
        const trans = bt.arr.transpose(perm);
        const y = new tensor_1.Tensor(trans.shape, x.dtype, null, new NdarrayTensor(trans));
        return y;
    }
    matMul(a, b, transposeA = false, transposeB = false) {
        const activeA = (transposeA) ? this.transpose(a, [1, 0]) : a;
        const activeB = (transposeB) ? this.transpose(b, [1, 0]) : b;
        const backA = activeA.backendTensor.arr;
        const backB = activeB.backendTensor.arr;
        const C = new tensor_1.Tensor([backA.shape[0], backB.shape[1]], activeB.dtype, null, null);
        const backC = C.backendTensor.arr;
        ndarray_gemm_1.default(backC, a, b, 1, 1);
        return C;
    }
    add(a, b) {
        const backA = a.backendTensor.arr;
        const backB = b.backendTensor.arr;
        const C = new tensor_1.Tensor(a.shape, a.dtype);
        const backC = C.backendTensor.arr;
        ndarray_ops_1.default.add(backC, backA, backB);
        return C;
    }
    neg(x) {
        const backX = x.backendTensor.arr;
        const y = new tensor_1.Tensor(x.shape, x.dtype);
        const backY = y.backendTensor.arr;
        ndarray_ops_1.default.neg(backY, backX);
        return y;
    }
    multiply(a, b) {
        const backA = a.backendTensor.arr;
        const backB = b.backendTensor.arr;
        const c = new tensor_1.Tensor(a.shape, a.dtype);
        const backC = c.backendTensor.arr;
        ndarray_ops_1.default.multiply(backC, backA, backB);
        return c;
    }
    relu(x) {
        const backX = x.backendTensor.arr;
        const y = new tensor_1.Tensor(x.shape, x.dtype);
        const backY = y.backendTensor.arr;
        ndarray_ops_1.default.maxs(backY, backX, 0);
        return y;
    }
}
const backendName = "JS_ndarray";
const backendScore = 2;
const backendJsNdarray = new JsNdarrayBackend();
environments_1.ENV.registerBackend(backendName, backendJsNdarray, backendScore);
//# sourceMappingURL=backend_js.js.map