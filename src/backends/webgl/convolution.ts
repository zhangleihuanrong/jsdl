import { NDView as NdArray } from '../../NdView/ndview';
import { WebGL2Driver } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";
import { GlslCodeUtil } from './glslCodeUtil';
import { assert as ASSERT } from '../../utils/gadget';
import { WebGlBaseProgram } from './program';

export class WebGlProgramConv2d extends WebGlBaseProgram {
    x: WebGLTensor;
    k: WebGLTensor;
    paddings: number[];
    strides: number[];
    dilations: number[];
    groups: number;
    bias: WebGLTensor;

    constructor(webgl:WebGL2Driver,
                x: WebGLTensor, k: WebGLTensor, bias: WebGLTensor,
                strides: number[], paddings: number[], dilations: number[], groups: number = 1) {
        super(webgl);

        this.x = x;
        this.k = k;
        this.paddings = paddings;
        this.strides = strides;
        this.dilations = dilations;
        this.groups = groups;
        this.bias = bias;
    }

    prepareOutput() {
        const [batchSize, inputChannels] = [this.x.shape[0], this.x.shape[1]];
        const rank = this.x.shape.length;
        const spatialRank = this.x.shape.length - 2;
        const inputSpatialShape = this.x.shape.slice(2);
        const inputSpatialShapeWithPad = inputSpatialShape.map((v, i) => v + this.paddings[i] + this.paddings[i + spatialRank]);
    
        ASSERT(this.k.shape.length === rank, 'Wrong kernel shape!');

        const [outChannels, kernelIn] = [this.k.shape[0], this.k.shape[1]];
        ASSERT(kernelIn * this.groups === inputChannels, 'Group dimension not match!');
        ASSERT(outChannels / this.groups === ((outChannels / this.groups) | 0), 'Can not grout out channels');
    
        ASSERT(this.bias == null || (this.bias.shape.length === 1 && this.bias.shape[0] === outChannels), 'Wrong bias shape');
        ASSERT(this.dilations.length === spatialRank, 'Wrong dilations dimension.');
        ASSERT(this.strides.length === spatialRank, 'Wrong strides dimension.');
        ASSERT(this.paddings.length === spatialRank * 2, 'Wrong pads dimension.');
    
        const kernelSpatialShape = this.k.shape.slice(2);
        const dilatedKernelShape = kernelSpatialShape.map((v, i) => v + (v - 1) * (this.dilations[i] - 1));
        const outputSpatialShape =
            inputSpatialShapeWithPad.map((v, i) => Math.floor((v - dilatedKernelShape[i] + this.strides[i]) / this.strides[i]));
        const outputShape = [batchSize, outChannels].concat(...outputSpatialShape);
    
        this.y = new WebGLTensor(new NdArray(null, outputShape), 'float32');
        this.y.PrepareGpuData(this.webgl);

        this.prgTextures = [{name: 'X', tensor: this.x}, {name: 'K', tensor: this.k}];
        if (this.bias) this.prgTextures.push({name: 'Bias', tensor: this.bias});

        this.prgUniforms = null;
    }


    generateFragShaderCode(): string {
        const rank = this.x.shape.length;
        const inputSpatialShape =this.x.shape.slice(2);
    
        const [outChannels, kernelIn] = [this.k.shape[0], this.k.shape[1]];
        const kernelSpatialShape = this.k.shape.slice(2);
        const groupOutChanels = outChannels / this.groups;
    
        const initValueSnippet = (this.bias == null) ? '0.0' : 'getBias(yC)';
        const codeLines: string[] = [];
        let indent = '    ';

        codeLines.push(`bool inpad_1 = false;`);
        for (let i = 2; i < rank; ++i) {
          codeLines.push(`${indent}int x_${i} = idx_${i} * ${this.strides[i - 2]} - ${this.paddings[i - 2]};`);
          codeLines.push(`${indent}bool inpad_${i} = inpad_${i - 1} || (x_${i} < 0 || x_${i} >= ${inputSpatialShape[i - 2]});`);
          codeLines.push(`${indent}for (int k_${i} = 0; k_${i} < ${kernelSpatialShape[i - 2]}; ++k_${i}) {`);
          indent += '    ';
        }
        codeLines.push(`${indent}for(int x_1 = xCStart; x_1 < xCLast; ++x_1) {`);
        codeLines.push(`${indent}    int k_1 = x_1;`);
        codeLines.push(`${indent}    float valX = (inpad_${rank-1}) ? 0.0 : getX(${GlslCodeUtil.argList(rank, 'x_')});`);
        codeLines.push(`${indent}    r += valX * getK(${GlslCodeUtil.argList(rank, 'k_')});`);
        codeLines.push(`${indent}}`);
    
        for (let i = rank - 1; i >= 2; --i) {
          codeLines.push(`${indent}x_${i} += ${this.dilations[i - 2]};`);
          codeLines.push(`${indent}inpad_${i} = inpad_${i-1} || (x_${i} < 0 || x_${i} >= ${inputSpatialShape[i - 2]});`);
          indent = indent.substring(4);
          codeLines.push(`${indent}}`);
        }
        const snippetMainLoop = codeLines.join('\n');

        const fsCode =  `${this.generateFragShaderHead('Convolution')}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(this.y, 'Y', 'idx_', 'outTex', '    ')}

    int batchId = idx_0;
    int yC = idx_1;

    int g = int(yC / ${groupOutChanels});
    int xCStart = g * ${kernelIn};
    int xCLast = xCStart + ${kernelIn};
    int x_0 = batchId;
    int k_0 = yC;

    float r = ${initValueSnippet};

    ${snippetMainLoop}

    outColor = vec4(r, 0.0, 0.0, 0.0);
}
`;
    return fsCode;
    }

//     conv2d(
//         X: WebGLTensor, 
//         K: WebGLTensor,
//         strides: [number, number],
//         dilations: [number, number],
//         groups: number = 1,
//         Bias: WebGLTensor = null) : WebGLTensor {

//         const [batchSize, inputRows, inputCols, inputChannels] = [X.shape[0], X.shape[1], X.shape[2], X.shape[3]];
//         const [kernelRows, kernelCols, kernelIn, outChannels] = [K.shape[0], K.shape[1], K.shape[2], K.shape[3]];
//         ASSERT(inputChannels == kernelIn * groups, "intput channel do not match kernnels*groups");
//         ASSERT(Math.floor(outChannels / groups) == outChannels / groups, "group can not equal devide outputChannels");
//         if (Bias) ASSERT(Bias.shape.length == 1 && Bias.shape[0] == outChannels, "bias shape are bad!")

//         const dilateKernelRows = kernelRows + (kernelRows - 1) * (dilations[0] - 1);
//         const dilateKernelCols = kernelCols + (kernelCols - 1) * (dilations[1] - 1);
//         const outRows = Math.floor((inputRows - dilateKernelRows + strides[0]) / strides[0]);
//         const outCols = Math.floor((inputCols - dilateKernelCols + strides[1]) / strides[1]);
//         const C = new WebGLTensor(new NdArray(null, [batchSize, outRows, outCols, outChannels]), 'float32');
//         C.PrepareGpuData(this.webgl);
//         const groupOutChanels = outChannels / groups;

//         const snippetDeclareBias = (Bias) ? `uniform smapler2D Bias;` : '';
//         const snippetAddBias = (Bias) ? `r = getBias(2, ['0', 'yC']);` : '';
//         const snippetGetBias = (Bias) ? GlslCodeUtil.glslGet(Bias, 'Bias') : '';


//         const code = `#version 300 es
// precision highp float;
// precision highp int;

// in vec2 outTex;
// uniform sampler2D X;
// uniform sampler2D K;
// ${snippetDeclareBias}
// out vec4 outColor;

// ${GlslCodeUtil.glslGet(X, 'X')}

// ${GlslCodeUtil.glslGet(K, 'K')}

// ${snippetGetBias}

// void main() {
//     ${GlslCodeUtil.snippetLogicFormST(C, 'C', ['batchId', 'yH', 'yW', 'yC'], 'outTex', '    ')}

//     int g = int(yC / ${groupOutChanels});  // channels per group = ${groupOutChanels}
//     int xCStart = g * ${kernelIn};
//     int xCLast = xCStart + ${kernelIn};

//     float r = 0.0;
//     ${snippetAddBias}
//     int xH = yH * ${strides[0]}; // strides[0] = ${strides[0]}
//     for (int kH = 0; kH < ${kernelRows}; ++kH) {
//         int xW = yW * ${strides[1]}; // strides[1] = ${strides[1]}
//         for (int kW = 0; kW < ${kernelCols}; ++kW) {
//             for (int xC = xCStart; xC < xCLast; ++xC) {
//                 float pixel = getX(batchId, xH, xW, xC);
//                 float fp = getK(kH, kW, xC, yC);
//                 r += (pixel * fp);
//             }
//             xW += ${dilations[1]};
//         }
//         xH += ${dilations[0]};
//     }
//     outColor = vec4(r, 0.0, 0.0, 0.0);
// }
// `;

//         //console.log(code);
//         const startCompile = new Date().getTime();
//         const program = this.webgl.compileProgram(code);
//         let msCompile = (new Date()).getTime() - startCompile;
//         console.log(`>>>>>>>>Compile glsl program cost ${msCompile}ms<<<<<<<<`);

//         const textures = [{ name: 'X', texture: X._texture }, { name: 'K', texture: K._texture }];
//         if (Bias) textures.push({ name: 'Bias', texture: Bias._texture });

//         this.webgl.runProgram(
//             program,
//             C._texture,
//             C._texShape,
//             textures, 
//             null);

//         return C;
//     };



};