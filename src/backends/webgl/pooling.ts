import { NDView as NdArray } from '../../NdView/ndview';
import { WebGL2Driver } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";
import { GlslCodeUtil } from './glslCodeUtil';
import { assert as ASSERT } from '../../utils/gadget';
import { WebGlBaseProgram } from './program';

export type PoolingType = 'max' | 'average';

// global max => set kernelShape as x's spatial shape, no padding, no strides
export class WebGlPoolingOp extends WebGlBaseProgram {
    poolingType: PoolingType;
    x: WebGLTensor;
    kernelShape: number[];
    paddings: number[];
    strides: number[];
    countIncludePadding: boolean = false;

    constructor(webgl:WebGL2Driver, poolingType: PoolingType,
                x: WebGLTensor, kernelShape: number[], 
                strides: number[], paddings: number[], countIncludePadding: boolean = false) {
        super(webgl);
        this.poolingType = poolingType;
        this.x = x;
        this.kernelShape = kernelShape;
        this.paddings = paddings;
        this.strides = strides;
        this.countIncludePadding = countIncludePadding;
    }

    prepareOutput() {
        const [batchSize, channels] = [this.x.shape[0], this.x.shape[1]];
        const spatialRank = this.x.shape.length - 2;
        const xSpatialShape = this.x.shape.slice(2);

        ASSERT(this.paddings.length === spatialRank * 2, 'Wrong pads dimension.');
        const paddedSpatialShape = xSpatialShape.map((v, i) => v + this.paddings[i] + this.paddings[i + spatialRank]);
    
        ASSERT(this.kernelShape.length === spatialRank, 'Wrong rank of kernel shape!');
        ASSERT(this.kernelShape.every((v, i) => (v > 0 && v < xSpatialShape[i])), "Kernel size value too big or negtive");
        ASSERT(this.strides.length === spatialRank, 'Wrong rank of strides.');
        ASSERT(this.strides.every((v, i) => (v > 0 && v < xSpatialShape[i])), "Kernel size value too big or negtive");
    
        const outputSpatialShape =
            paddedSpatialShape.map((v, i) => Math.floor((v - this.kernelShape[i] + this.strides[i]) / this.strides[i]));
        const outputShape = [batchSize, channels].concat(...outputSpatialShape);
    
        this.y = new WebGLTensor(new NdArray(null, outputShape), 'float32');
        this.y.PrepareGpuData(this.webgl);

        this.prgTextures = [{name: 'X', tensor: this.x}];
        this.prgUniforms = null;
    }


    generateFragShaderCode(): string {
        const rank = this.x.shape.length;
        const xSpatialShape = this.x.shape.slice(2);
        const kernelPointCount = this.kernelShape.reduce((m, v) => m * v, 1);

        const snippetOnZero = (this.poolingType == 'average') ? '' : 'if (r < 0.0) { r = 0.0; }';
        const snippetOnValue = (this.poolingType == 'average') ? 'r += valX;' : 'if (r < valX) { r = valX; }';

        const initValueSnippet = (this.poolingType == 'average') ? '0.0' : 'float(-1.0 / 0.0)';
        const codeLines: string[] = [];
        let indent = '    ';

        for (let i = 2; i < rank; ++i) {
          codeLines.push(`${indent}int x_${i}B = y_${i} * ${this.strides[i-2]} - ${this.paddings[i-2]};`);
          codeLines.push(`${indent}int x_${i}E = x_${i}B + ${this.kernelShape[i-2]} * ${this.strides[i-2]};`);
          codeLines.push(`${indent}if (x_${i}B < 0) {`);
          codeLines.push(`${indent}    ${snippetOnZero};`);
          codeLines.push(`${indent}    x_${i}B += int(ceil(float(0 - x_${i}B) / float(${this.strides[i-2]}))) * ${this.strides[i-2]};`);
          codeLines.push(`${indent}}`);
          codeLines.push(`${indent}if (x_${i}E - ${this.strides[i-2]} >= ${xSpatialShape[i-2]}) {`);
          codeLines.push(`${indent}    ${snippetOnZero}`);
          codeLines.push(`${indent}    x_${i}E -= int(ceil(float(x_${i}E - ${this.kernelShape[i-2]}) / float(${this.strides[i-2]}))) * ${this.strides[i-2]};`);
          codeLines.push(`${indent}}`);
          codeLines.push(`${indent}count *= int((x_${i}E - x_${i}B) / ${this.strides[i-2]});`);
          codeLines.push(`${indent}for (int x_${i} = x_${i}B; x_${i} < x_${i}E; x_${i} += ${this.strides[i-2]}) {`);
          indent += '    ';
        }
        codeLines.push(`${indent}float valX = getX(${GlslCodeUtil.argList(rank, 'x_')});`);
        codeLines.push(`${indent}${snippetOnValue}`);
    
        for (let i = rank - 1; i >= 2; --i) {
          indent = indent.substring(4);
          codeLines.push(`${indent}}`);
        }
        const snippetMainLoop = codeLines.join('\n');

        const avgSnippet = (this.poolingType == 'max') ? '' : 
                    (this.countIncludePadding) ? `r = r / float(${kernelPointCount});` : `r = r / float(count);`;
        const fsCode =  `${this.generateFragShaderHead(this.poolingType + 'Pooling')}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(this.y, 'Y', 'y_', 'outTex', '    ')}

    int x_0 = y_0;
    int x_1 = y_1;

    float r = ${initValueSnippet};
    int count = 1;

    ${snippetMainLoop}

    ${avgSnippet}

    outColor = vec4(r, 0.0, 0.0, 0.0);
}
`;
    return fsCode;
    }

};
