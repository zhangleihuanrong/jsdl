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

};