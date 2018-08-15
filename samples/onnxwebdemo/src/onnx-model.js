'use strict';

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const protobuf = require('protobufjs');
const tf = require('@tensorflow/tfjs');

eval(fs.readFileSync(path.join(__dirname, 'onnx.js')) + ' ');

const onnx = protobuf.roots.onnx.onnx;

function readModelBuffer(buffer) {
    return onnx.ModelProto.decode(buffer);
}

exports.readModelBuffer = readModelBuffer;

exports.readModeFile = function(modelFile) {
    let buffer = fs.readFileSync(modelFile);
    return readModelBuffer(buffer);
};

function executeNode(predFlow, node) {
    if (node.opType == "Conv") {
        const nameX = node.input[0];
        let tensorX = predFlow.tensors[nameX];
        assert(tensorX.shape.length == 4); // ensure conv2d

        tensorX = tf.transpose(tensorX, [0, 2, 3, 1]); // make it NHWC mode

        const nameW = node.input[1];
        let tensorW = predFlow.tensors[nameW];
        tensorW = tf.transpose(tensorW, [2, 3, 1, 0]); // make it [H, W, in, out] mode

        const attrs = node.attribute;
        let pads = attrs.find(attr => attr.name == 'pads');
        if (pads) {
            pads = pads.ints.map(lv => Number(lv));
            if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
                const tfpad = [[0, 0], [pads[0], pads[2]], [pads[1], pads[3]], [0, 0]];
                tensorX = tf.pad(tensorX, tfpad, 0);
            }
        }

        const strides = attrs.find(attr => attr.name == 'strides').ints.map(lv => Number(lv));
        let tensorY = tf.conv2d(tensorX, tensorW, strides, 'valid', 'NHWC');
        tensorY = tf.transpose(tensorY, [0, 3, 1, 2]); // make it NCHW

        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else if (node.opType == "MaxPool" || node.opType == "AveragePool" ) {
        const nameX = node.input[0];
        let tensorX = predFlow.tensors[nameX];
        tensorX = tf.transpose(tensorX, [0, 2, 3, 1]); // make it NHWC mode
        const attrs = node.attribute;
        let pads = attrs.find(attr => attr.name == 'pads');
        if (pads) {
            pads = pads.ints.map(lv => Number(lv));
            if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
                const tfpad = [[0, 0], [pads[0], pads[2]], [pads[1], pads[3]], [0, 0]];
                tensorX = tf.pad(tensorX, tfpad, 0);
            }
        }

        const strides = attrs.find(attr => attr.name == 'strides').ints.map(lv => Number(lv));
        const filterSize = attrs.find(attr => attr.name == 'kernel_shape').ints.map(lv => Number(lv));
        
        let tensorY = (node.opType == "MaxPool") ? 
            tf.maxPool(tensorX, filterSize, strides, 'valid') : 
            tf.avgPool(tensorX, filterSize, strides, 'valid') ;
        tensorY = tf.transpose(tensorY, [0, 3, 1, 2]); // make it NCHW

        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else if (node.opType == "Relu") {
        const nameX = node.input[0];
        const tensorX = predFlow.tensors[nameX];
        const tensorY = tf.relu(tensorX);
        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else if (node.opType == "BatchNormalization") {
        const nameX = node.input[0];
        let tensorX = predFlow.tensors[nameX];
        tensorX = tf.transpose(tensorX, [0, 2, 3, 1]);

        const nameScale = node.input[1];
        const tensorScale = predFlow.tensors[nameScale];
        const nameB = node.input[2];
        const tensorB = predFlow.tensors[nameB];
        const nameMean = node.input[3];
        const tensorMean = predFlow.tensors[nameMean];
        const nameVar = node.input[4];
        const tensorVar = predFlow.tensors[nameVar];
        const epsilon = node.attribute.find(attr => attr.name == 'epsilon').f;

        let tensorY = tf.batchNormalization(tensorX, tensorMean, tensorVar, epsilon, tensorScale, tensorB);
        tensorY = tf.transpose(tensorY, [0, 3, 1, 2]);
        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else if (node.opType == "Sum") {
        const nameX = node.input[0];
        let tensorY = predFlow.tensors[nameX];
        for (let i = 1; i < node.input.length; ++i) {
            const nameX = node.input[i];
            let tensorX = predFlow.tensors[nameX];
            tensorY = tf.add(tensorY, tensorX);
        }
        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else if (node.opType == "Gemm") {
        const nameA = node.input[0];
        let tensorA = predFlow.tensors[nameA];
        const axises = [];
        for (let i = 2; i < tensorA.shape.length; ++i) {
            axises.push(i);
        }
        tensorA = tf.squeeze(tensorA, axises);

        const nameB = node.input[1];
        let tensorB = predFlow.tensors[nameB];
        const nameC = node.input[2];
        const tensorC = predFlow.tensors[nameC];

        const transA = node.attribute.find(attr => attr.name == 'transA') ? node.attribute.find(attr => attr.name == 'transA').i : 0;
        const transB = node.attribute.find(attr => attr.name == 'transB') ? node.attribute.find(attr => attr.name == 'transB').i : 0;

        let tensorY = tf.matMul(tensorA, tensorB, transA != 0, transB != 0);
        tensorY = tf.add(tensorY, tensorC);

        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else if (node.opType == "Softmax") {
        const nameX = node.input[0];
        const tensorX = predFlow.tensors[nameX];
        const tensorY = tf.softmax(tensorX);
        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else if (node.opType == "Reshape") {
        const nameX = node.input[0];
        const tensorX = predFlow.tensors[nameX];
        const nameShape = node.input[1];
        const tensorShape = predFlow.tensors[nameShape];
        const tensorY = tf.reshape(tensorX, tensorShape.dataSync());
        const nameY = node.output[0];
        predFlow.tensors[nameY] = tensorY;
    }
    else {
        throw new Error(`operator not supported currently: ${node.opType}`);
    }
}
    
function predict(onnxModel, inputArray)  {
    const predFlow = {};
    predFlow.name = onnxModel.graph.name;
    predFlow.initialValues = {};
    predFlow.tensors = {};

    onnxModel.graph.initializer.map(initTensor => {
        predFlow.initialValues[initTensor.name] = initTensor;
    });

    const strangeInit = { };
    onnxModel.graph.input.map(ti => {
        assert (ti.type.tensorType);

        const name =  ti.name;
        const initValue = predFlow.initialValues[name];
        if (initValue) {
            const buf = initValue.rawData.buffer.slice(initValue.rawData.byteOffset, initValue.rawData.byteOffset + initValue.rawData.byteLength);
            const shape = initValue.dims.map(lv => Number(`${lv}`));
            if (ti.type.tensorType.elemType == 1) {
                const floatsArray = new Float32Array(buf);
                const ts = tf.tensor(floatsArray, shape, 'float32');
                predFlow.tensors[name] = ts;
            }
            else if (ti.type.tensorType.elemType == 7) {
                const ba = new Uint8Array(buf);
                const floatsArray = []; //new Number[buf.byteLength/8];
                for (let i = 0; i < ba.length/8; i++) {
                    let value = 0.0;
                    const k = i * 8;
                    let weight = 1;
                    for (let j = 0; j < 6; ++j) {
                        //only handle positive
                        value = value + ba[k + j] * weight;
                        weight *= 256.0;
                    }
                    floatsArray[i] = value;
                }
                //const floatsArray = /*(ti.type.tensorType.elemType == 1) */ new Float32Array(buf);
                const ts = tf.tensor(floatsArray, shape, 'float32');
                predFlow.tensors[name] = ts;
            }
        }
        else {
            const shape = ti.type.tensorType.shape.dim.map(dv => Number(dv.dimValue));
            // const ts = tf.input({shape});
            const ts = tf.tensor(inputArray, shape, 'float32')
            predFlow.tensors[name] = ts;
        }
    });

    console.log(`  ==== Start executing model...`);
    onnxModel.graph.node.map(node => {
        executeNode(predFlow, node);
    });
    console.log(`  ==== Finished executing model...`);

    const outputs = onnxModel.graph.output.map(tout => {
        assert(tout.type.tensorType);
        const name =  tout.name;
        return predFlow.tensors[name];
    });

    return outputs;
}

exports.predict = predict;

function testBatchNormalization() {
    let x = tf.tensor4d([[[[-1, 0, 1]], [[2, 3, 4]]]]); //[1,2,1,3]
    const s = tf.tensor1d([1.0, 1.5]);
    const b = tf.tensor1d([0.0, 1.5]);
    const mean = tf.tensor1d([0.0, 3.0]);
    const v = tf.tensor1d([1.0, 1.5]);

    x = tf.transpose(x, [0, 2, 3, 1]);
    x.print();

    let y = tf.batchNormalization(x, mean, v, 1e-5, s, b);

    y.print();
}

//testBatchNormalization();
