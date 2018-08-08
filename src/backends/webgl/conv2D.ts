import { NDView as NdArray } from '../../NdView/ndview';
import { WebGL2Driver } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";
import { CoordinateMapping } from './coord2D';
import { assert as ASSERT } from '../../utils/gadget';

export class WebGlProgramConv2d {
    webgl: WebGL2Driver;
    x: WebGLTensor;
    k: WebGLTensor;
    strides: [number, number];
    dilations: [number, number];
    groups: number;
    bias: WebGLTensor;

    constructor(
        webgl:WebGL2Driver,
        x: WebGLTensor, 
        k: WebGLTensor,
        strides: [number, number],
        dilations: [number, number],
        groups: number = 1,
        bias: WebGLTensor = null) {
            this.webgl = webgl;
            this.x = x;
            this.k = k;
            this.strides = strides;
            this.dilations = dilations;
            this.groups = groups;
            this.bias = bias;
    }


    conv2d(
        X: WebGLTensor, 
        K: WebGLTensor,
        strides: [number, number],
        dilations: [number, number],
        groups: number = 1,
        Bias: WebGLTensor = null) : WebGLTensor {

        const [batchSize, inputRows, inputCols, inputChannels] = [X.shape[0], X.shape[1], X.shape[2], X.shape[3]];
        const [kernelRows, kernelCols, kernelIn, outChannels] = [K.shape[0], K.shape[1], K.shape[2], K.shape[3]];
        ASSERT(inputChannels == kernelIn * groups, "intput channel do not match kernnels*groups");
        ASSERT(Math.floor(outChannels / groups) == outChannels / groups, "group can not equal devide outputChannels");
        if (Bias) ASSERT(Bias.shape.length == 1 && Bias.shape[0] == outChannels, "bias shape are bad!")

        const dilateKernelRows = kernelRows + (kernelRows - 1) * (dilations[0] - 1);
        const dilateKernelCols = kernelCols + (kernelCols - 1) * (dilations[1] - 1);
        const outRows = Math.floor((inputRows - dilateKernelRows + strides[0]) / strides[0]);
        const outCols = Math.floor((inputCols - dilateKernelCols + strides[1]) / strides[1]);
        const C = new WebGLTensor(new NdArray(null, [batchSize, outRows, outCols, outChannels]), 'float32');
        C.PrepareGpuData(this.webgl);
        const groupOutChanels = outChannels / groups;

        const snippetDeclareBias = (Bias) ? `uniform smapler2D Bias;` : '';
        const snippetAddBias = (Bias) ? `r = getBias(2, ['0', 'yC']);` : '';
        const snippetGetBias = (Bias) ? CoordinateMapping.glslGet(Bias, 'Bias') : '';


        const code = `#version 300 es
precision highp float;
precision highp int;

in vec2 outTex;
uniform sampler2D X;
uniform sampler2D K;
${snippetDeclareBias}
out vec4 outColor;

${CoordinateMapping.glslGet(X, 'X')}

${CoordinateMapping.glslGet(K, 'K')}

${snippetGetBias}

void main() {
    ${CoordinateMapping.snippetLogicFormST(C, 'C', ['batchId', 'yH', 'yW', 'yC'], 'outTex', '    ')}

    int g = int(yC / ${groupOutChanels});  // channels per group = ${groupOutChanels}
    int xCStart = g * ${groupOutChanels};
    int xCLast = xCStart + ${groupOutChanels};

    float r = 0.0;
    ${snippetAddBias}
    int xH = yH * ${strides[0]}; // strides[0] = ${strides[0]}
    for (int kH = 0; kH < ${kernelRows}; ++kH) {
        int xW = yW * ${strides[1]}; // strides[1] = ${strides[1]}
        for (int kW = 0; kW < ${kernelCols}; ++kW) {
            for (int xC = xCStart; xC < xCLast; ++xC) {
                float pixel = getX(batchId, xH, xW, xC);
                float fp = getK(kH, kW, xC, yC);
                r += (pixel * fp);
            }
            xW += ${dilations[1]};
        }
        xH += ${dilations[0]};
    }
    outColor = vec4(r, 0.0, 0.0, 0.0);
}
`;

        //console.log(code);
        const startCompile = new Date().getTime();
        const program = this.webgl.compileProgram(code);
        let msCompile = (new Date()).getTime() - startCompile;
        console.log(`>>>>>>>>Compile glsl program cost ${msCompile}ms<<<<<<<<`);

        const textures = [{ name: 'X', texture: X._texture }, { name: 'K', texture: K._texture }];
        if (Bias) textures.push({ name: 'Bias', texture: Bias._texture });

        this.webgl.runProgram(
            program,
            C._texture,
            C._texShape,
            textures, 
            null);

        return C;
    };

    run(): WebGLTensor {
        return this.conv2d(this.x, this.k, this.strides, this.dilations, this.groups, this.bias);
    }
};