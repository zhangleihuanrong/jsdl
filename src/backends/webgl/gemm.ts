import { WebGLTensor } from "../backend_webgl";
import { WebGL2Driver } from "./webgl2";
import { GlslCodeUtil } from './glslCodeUtil';

import { NDView as NdArray } from '../../NdView/ndview';
import { assert as ASSERT } from '../../utils/gadget';
import { WebGlBaseProgram } from "./program";
import { WebGLBinaryOp } from "./arithmetic";


// return A * B * alpha + bias * beta, where A=(transposeA)?a':a, silimar to B
// A => [M, K], B: [K, N], bias: castable[M, N]
// gemm(a: WebGLTensor, b: WebGLTensor, bias: WebGLTensor, alpha: number, beta: number, transposeA: boolean, transposeB: boolean) : Tensor;
export class WebGlGemmOp extends WebGlBaseProgram {
    a: WebGLTensor;
    b: WebGLTensor;
    alpha: number = 1.0;
    bias: WebGLTensor = null;

    constructor(webgl:WebGL2Driver, 
                a: WebGLTensor, b: WebGLTensor, bias: WebGLTensor, 
                alpha: number, beta: number, transposeA: boolean, transposeB: boolean) {
        super(webgl);
        this.a = (transposeA) ? a.transpose() : a;
        this.b = (transposeB) ? b.transpose() : b;
        this.alpha = alpha;
        this.bias = bias;
        if (bias && beta !== 1) {
            ASSERT(bias.shape.length === 1 && bias.shape[0] == b.shape[1], 'Wrong bias shpae');
            const scalaT = new WebGLTensor(new NdArray(new Float32Array([alpha]), [1]), 'float32');
            scalaT.MoveDataToGpu(this.webgl);
            const mulResult = new WebGLBinaryOp(this.webgl, 'mul', this.bias, scalaT, true);
            this.bias = mulResult.run();
        }
    }

    prepareOutput() {
        ASSERT(this.a.shape.length === 2,  'Wrong rank for a');
        ASSERT(this.b.shape.length === 2,  'Wrong rank for b');
        ASSERT(this.a.shape[1] == this.b.shape[0], 'Common dimension could not be matched');
        ASSERT(this.bias == null || (this.bias.shape.length === 1 && this.bias.shape[0] == this.b.shape[1]), 'Wrong bias shpae');
        const M = this.a.shape[0];
        const N = this.b.shape[1];

        this.y = new WebGLTensor(new NdArray(null, [M, N]), 'float32');
        this.y.PrepareGpuData(this.webgl);
        this.prgTextures = [
            {name: 'A', tensor: this.a},
            {name: 'B', tensor: this.b},
        ];
        if (this.bias) {
            this.prgTextures.push({name: 'Bias', tensor: this.bias});
        }
    }

    generateFragShaderCode(): string {
        const commonDim = this.a.shape[1];
        const initBiasSnippet = (this.bias) ? `getBias(y_1)` : `0.0`;

        const fsCode =  
`${this.generateFragShaderHead('Batch Normalize Fragment Shader')}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(this.y, 'Y', 'y_', 'outTex', '    ')}

    float sum = ${initBiasSnippet};
    for (int k = 0; k < ${commonDim}; ++k) {  // length of the common axis
        sum += float(${this.alpha}) * float(getA(y_0, k)) * float(getB(k, y_1));
    }
    outColor = vec4(sum, 0.0, 0.0, 0.0);
}
`;

        console.log(fsCode);
        return fsCode;
    }

};