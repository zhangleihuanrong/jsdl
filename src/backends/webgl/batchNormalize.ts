import { NDView as NdArray } from '../../NdView/ndview';
import { WebGL2Driver } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";
import { GlslCodeUtil } from './glslCodeUtil';
import { assert as ASSERT } from '../../utils/gadget';
import { WebGlBaseProgram } from './program';
import { WebGlProgramUnaryOp } from './unaryops';
import { WebGLBinaryOp } from './arithmetic';

export class WebGlOpBatchNormalize extends WebGlBaseProgram {
    x: WebGLTensor;
    scale: WebGLTensor;
    bias: WebGLTensor;
    mean: WebGLTensor;
    variance: WebGLTensor;
    epsilon : number;
    
    // intermediate
    sqrtVar: WebGLTensor;

    // x: [N, C, ......], other tensor are of [C]
    constructor(webgl:WebGL2Driver, x: WebGLTensor,
                scale: WebGLTensor, bias: WebGLTensor, mean: WebGLTensor, variance: WebGLTensor, 
                epsilon: number, momentum?: number, spatial?: number) {
        super(webgl);

        ASSERT(x.shape.length >= 2, "input size should be [N, C, ...]");
        const C = x.shape[1];
        ASSERT(scale.shape.length == 1 && scale.shape[0] == C, "wrong shape for scale");
        ASSERT(bias.shape.length == 1 && bias.shape[0] == C, "wrong shape for bias");
        ASSERT(mean.shape.length == 1 && mean.shape[0] == C, "wrong shape for mean");
        ASSERT(variance.shape.length == 1 && variance.shape[0] == C, "wrong shape for variance");

        this.x = x;
        this.scale = scale;
        this.bias = bias;
        this. mean = mean;
        this.variance = variance;
        this.epsilon = epsilon;
    }

    prepareOutput() {
        const sqrtOp = new WebGlProgramUnaryOp(this.webgl, 'sqrt', this.variance);
        this.sqrtVar = sqrtOp.run();
        const fa = new Float32Array([this.epsilon]);
        const epsilonT = new WebGLTensor(new NdArray(fa, [1]), 'float32');
        const maxOp = new WebGLBinaryOp(this.webgl, 'max', this.sqrtVar, epsilonT.MoveDataToGpu(this.webgl));
        this.sqrtVar = maxOp.run();

        this.y = new WebGLTensor(new NdArray(null, this.x.shape), this.x._dtype);
        this.y.PrepareGpuData(this.webgl);
        this.prgTextures = [
            {name: 'X', tensor: this.x},
            {name: 'S', tensor: this.scale},
            {name: 'B', tensor: this.bias},
            {name: 'M', tensor: this.mean},
            {name: 'V', tensor: this.sqrtVar},
        ];
    }

    generateFragShaderCode(): string {
        const rankX = this.x.shape.length;

        const fsCode =  
`${this.generateFragShaderHead('Batch Normalize Fragment Shader')}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(this.y, 'Y', 'y_', 'outTex', '    ')}

    float val_x = getX(${GlslCodeUtil.argList(rankX, 'y_')});
    float val_scale = getS(y_1);
    float val_bias = getB(y_1);
    float val_mean = getM(y_1);
    float val_variance = getV(y_1);
    
    //((x - mean) / squareRoot(variance)) * scale + bias
    float r = (val_x - val_mean) / val_variance * val_scale + val_bias;
    outColor = vec4(r, 0.0, 0.0, 0.0);
}
`;

        console.log(fsCode);
        return fsCode;
    }
};
