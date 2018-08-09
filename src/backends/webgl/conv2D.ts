import { NDView as NdArray } from '../../NdView/ndview';
import { WebGL2Driver, UniformParameter } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";
import { CoordinateMapping } from './coord2D';
import { assert as ASSERT } from '../../utils/gadget';

export class WebGlProgramConv2d {
    webgl: WebGL2Driver;
    x: WebGLTensor;
    k: WebGLTensor;
    paddings: number[];
    strides: [number, number];
    dilations: [number, number];
    groups: number;
    bias: WebGLTensor;
    prg : WebGLProgram;

    constructor(
            webgl:WebGL2Driver,
            x: WebGLTensor, 
            k: WebGLTensor,
            paddings: number[], // [HPadBefore, WPadBefore, ..]
            strides: [number, number],
            dilations: [number, number],
            groups: number = 1,
            bias: WebGLTensor = null) {

        this.webgl = webgl;
        this.x = x;
        this.k = k;
        this.paddings = paddings;
        this.strides = strides;
        this.dilations = dilations;
        this.groups = groups;
        this.bias = bias;

        let programKeyName = 'Conv2d_General';
        if (x._array.isCoreOnly && k._array.isCoreOnly && (!bias || bias._array.isCoreOnly)) {
            programKeyName = 'Conv2d_CoreCoreCore';
        }
        this.prg = this.webgl.getProgram(programKeyName);
        if (this.prg == null) {
            this.prg = (programKeyName == 'Conv2d_CoreCoreCore')? this.buildCoreProgram() : this.buildGeneralProgram();
            this.webgl.setProgram(programKeyName, this.prg);
        }
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

    run() {
        let X = this.x;
        let K = this.k;
        let P = this.paddings;
        const [batchSize, inputRows, inputCols, inputChannels] = [X.shape[0], X.shape[1] + P[0] + P[2], X.shape[2] + P[1] + P[3], X.shape[3]];
        const [kernelRows, kernelCols, kernelIn, outChannels] = [K.shape[0], K.shape[1], K.shape[2], K.shape[3]];
        ASSERT(inputChannels == kernelIn * this.groups, "intput channel do not match kernnels*groups");
        ASSERT(Math.floor(outChannels / this.groups) == outChannels / this.groups, "group can not equal devide outputChannels");
        if (this.bias) ASSERT(this.bias.shape.length == 1 && this.bias.shape[0] == outChannels, "bias shape are bad!")

        const dilateKernelRows = kernelRows + (kernelRows - 1) * (this.dilations[0] - 1);
        const dilateKernelCols = kernelCols + (kernelCols - 1) * (this.dilations[1] - 1);
        const outRows = Math.floor((inputRows - dilateKernelRows + this.strides[0]) / this.strides[0]);
        const outCols = Math.floor((inputCols - dilateKernelCols + this.strides[1]) / this.strides[1]);
        const C = new WebGLTensor(new NdArray(null, [batchSize, outRows, outCols, outChannels]), 'float32');
        C.PrepareGpuData(this.webgl);

        const textures = [{ name: 'X', texture: X._texture }, { name: 'K', texture: K._texture }];
        if (this.bias) textures.push({ name: 'Bias', texture: this.bias._texture });

        const uniforms : UniformParameter[] = [
            { name: 'XShape', dtype : 'int32', value: this.x._array.coreShape},
            { name: 'XStride', dtype : 'int32', value: this.x._array.coreStride},
            { name: 'XOffsetTexWTexY', dtype : 'int32', value: [this.x._array.coreOffset, this.x._texShape[0], this.x._texShape[1]]},
            { name: 'KShape', dtype : 'int32', value: this.k._array.coreShape},
            { name: 'KStride', dtype : 'int32', value: this.k._array.coreStride},
            { name: 'KOffsetTexWTexY', dtype : 'int32', value: [this.k._array.coreOffset, this.k._texShape[0], this.k._texShape[1]]},
            { name: 'CShape', dtype : 'int32', value: C._array.coreShape},
            { name: 'CStride', dtype : 'int32', value: C._array.coreStride},
            { name: 'COffsetTexWTexY', dtype : 'int32', value: [C._array.coreOffset, C._texShape[0], C._texShape[1]]},
            { name: 'UseBias', dtype : 'int32', value: (this.bias)? 1 : 0},
            { name: 'strides', dtype : 'int32', value: this.strides},
            { name: 'dilations', dtype : 'int32', value: this.dilations},
            { name: 'paddings', dtype : 'int32', value: this.paddings},
            { name: 'groups', dtype : 'int32', value: this.groups},
        ];

        if (this.bias) {
            uniforms.push(
                { name: 'BiasShape', dtype : 'int32', value: this.bias._array.coreShape},
                { name: 'BiaStride', dtype : 'int32', value: this.bias._array.coreStride},
                { name: 'BiasOffsetTexWTexY', dtype : 'int32', value: [this.bias._array.coreOffset, this.x._texShape[0], this.x._texShape[1]]}
            );
        }

        this.webgl.runProgram(
            this.prg,
            C._texture,
            C._texShape,
            textures, 
            uniforms);


        return C;
    }

    private buildCoreProgram(){

        const code = `#version 300 es
precision highp float;
precision highp int;

in vec2 outTex;

uniform sampler2D X;
uniform sampler2D K;
//uniform sampler2D Bias;

uniform int XShape[4];          //BHWC
uniform int XStride[4];
uniform int XOffsetTexWTexY[3];

uniform int KShape[4];
uniform int KStride[4];
uniform int KOffsetTexWTexY[3];

uniform int UseBias;
// uniform int BiasShape[1];
// uniform int BiasStride[1];
// uniform int BiasOffsetTexWTexY[3];

uniform int CShape[4];
uniform int CStride[4];
uniform int COffsetTexWTexY[3];

uniform int strides[2];
uniform int dilations[2];
uniform int paddings[4]; //[HPadBefore, WPadBefore, HPadAfter, WPadAfter]
uniform int groups;

out vec4 outColor;

${CoordinateMapping.glslGetCoreOnly(4, 'X', 'float32')}

${CoordinateMapping.glslGetCoreOnly(4, 'K', 'float32')}

//$ {CoordinateMapping.glslGetCoreOnly(1, 'Bias', 'float32')}

float getPaddedX(int batchId, int xHP, int xWP, int xC) {
    if (xHP < paddings[0] || xHP >= (paddings[0] + XShape[1]) || xWP < paddings[1] || xWP >=  (paddings[1] + XShape[2])) {
        return 0.0;
    }
    return getX(batchId, xHP - paddings[0], xWP - paddings[1], xC);
}

void main() {
    ${CoordinateMapping.generalOutputIndexFormST(4, 'C', ['batchId', 'yH', 'yW', 'yC'], 'outTex', '    ')}

    int groupOutChanels = int(KShape[3] / groups); // channels per group
    int g = int(yC / groupOutChanels);  
    int xCStart = g * groupOutChanels;
    int xCLast = xCStart + groupOutChanels;

    float r = 0.0;
    // if (UseBias != 0) {
    //     r = getBias(yC);
    // }

    int xH = yH * strides[0];
    for (int kH = 0; kH < KShape[0]; ++kH) { // kernelRows = KShape[0]
        int xW = yW * strides[1];
        for (int kW = 0; kW < KShape[1]; ++kW) {  //kernelCols = KShape[1]
            for (int xC = xCStart; xC < xCLast; ++xC) {
                float pixel = getPaddedX(batchId, xH, xW, xC);
                float fp = getK(kH, kW, xC, yC);
                r += (pixel * fp);
            }
            xW += dilations[1];
        }
        xH += dilations[0];
    }
    outColor = vec4(r, 0.0, 0.0, 0.0);
}
`;

        console.log(code);
        const startCompile = new Date().getTime();
        const program = this.webgl.compileProgram(code);
        let msCompile = (new Date()).getTime() - startCompile;
        console.log(`>>>>>>>>Compile glsl program cost ${msCompile}ms<<<<<<<<`);
        return program;
    }

    private buildGeneralProgram(): WebGLProgram {
        return null;
    }

};