import { WebGLTensor } from "../backend_webgl";
import { WebGL2Driver } from "./webgl2";
import { GlslCodeUtil } from './glslCodeUtil';

import { NDView as NdArray } from '../../NdView/ndview';
import { assert as ASSERT } from '../../utils/gadget';
import { canBroadcastTo, getUnsqueezeAxisForBroadcast, getUnsqueezedShapeForBroadcast, getBroadcastRepeats } from '../../utils/broadcast';


export interface WebGlMatMulParameters {
    A: WebGLTensor;
    B: WebGLTensor;
    transposeA?: boolean;
    transposeB?: boolean;
};

export class WebGlProgramMatMul {
    webgl: WebGL2Driver;
    args: WebGlMatMulParameters;

    constructor(webgl:WebGL2Driver, args: WebGlMatMulParameters) {
        this.webgl = webgl;
        this.args = args;
    }

    run(): WebGLTensor {
        return this.matMul(this.args.A, this.args.B, this.args.transposeA, this.args.transposeB);
    }

    // output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.
    matMul(A: WebGLTensor, B: WebGLTensor, transposeA?: boolean, transposeB?: boolean): WebGLTensor {
        ASSERT(A.shape.length >= 1 && B.shape.length >= 1, "Scala is not allowed in matMul");
        ASSERT(A.shape.length >= B.shape.length, "Too many dimensions on b");

        if (transposeA == true && A.shape.length > 1) A = A.transpose();
        if (transposeB == true && B.shape.length > 1) B = B.transpose();

        if (A.shape.length == 1) A = A.expandDim(0);
        if (B.shape.length == 1) B = B.expandDim(1);

        const shapeAMul = A.shape.slice(A.shape.length - 2);
        const shapeBMul = B.shape.slice(B.shape.length - 2);
        ASSERT(shapeAMul[1] == shapeBMul[0], `shape[${shapeAMul}] can not matMul with shape[${shapeBMul}]`);
        const commonDim = shapeAMul[1];

        const shapeAPrefix = A.shape.slice(0, A.shape.length - 2);
        const shapeBPrefix = B.shape.slice(0, B.shape.length - 2);
        if (shapeAPrefix.length > 0) {
            ASSERT(canBroadcastTo(shapeAPrefix, shapeBPrefix), "Can not broadcast b to a");
            const unsq = getUnsqueezeAxisForBroadcast(shapeAPrefix, shapeBPrefix);
            if (unsq) B = B.unsqueeze(unsq);
            const shapeBExpanded = getUnsqueezedShapeForBroadcast(shapeAPrefix, shapeBPrefix);
            const repeats = getBroadcastRepeats(shapeAPrefix, shapeBExpanded);
            if (repeats) B = B.tile(repeats.concat(1, 1));
        }

        const shapeC = shapeAPrefix.concat(shapeAMul[0], shapeBMul[1]);
        const rankC = shapeC.length;
        const C = new WebGLTensor(new NdArray(null, shapeC)).PrepareGpuData(this.webgl);

        const code = `#version 300 es
precision highp float;
precision highp int;

in vec2 outTex;
uniform sampler2D A;
uniform sampler2D B;
out vec4 outColor;

${GlslCodeUtil.glslGet(A, 'A')}

${GlslCodeUtil.glslGet(B, 'B')}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(C, 'C', 'idx_', 'outTex', '    ')}

    float sum = 0.0;
    for (int k = 0; k < ${commonDim}; ++k) {  // length of the common axis
        float a = getA(${GlslCodeUtil.argList(rankC, 'idx_', [rankC-1, 'k'])});
        float b = getB(${GlslCodeUtil.argList(rankC, 'idx_', [rankC-2, 'k'])});
        sum += (a * b);
    }

    outColor = vec4(sum, 0.0, 0.0, 0.0);
}
`;
 
        //console.log(code);
        const startCompile = new Date().getTime();
        const program = this.webgl.compileProgram(code);
        let msCompile = (new Date()).getTime() - startCompile;
        console.log(`>>>>>>>>Compile glsl program cost ${msCompile}ms<<<<<<<<`);

        this.webgl.runProgram(
            program,
            C._texture,
            C._texShape,
            [{ name: 'A', tensor: A}, { name: 'B', tensor: B}], 
            null);

        return C;
    }

};