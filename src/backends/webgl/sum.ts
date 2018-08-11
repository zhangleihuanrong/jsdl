import { WebGLTensor } from "../backend_webgl";
import { WebGL2Driver } from "./webgl2";
import { GlslCodeUtil } from './glslCodeUtil';

import { NDView as NdArray } from '../../NdView/ndview';
import { assert as ASSERT, simpleHash32 } from '../../utils/gadget';
//import { DataType } from "../../types";

// Before call this, you need to do 
// => transpose, 
// => rebuild (if needed, construct new texture)
// => reshape,( get a clean tensor2d [batchSize, N] )
// => reduce on the N => [batchSize, 1]
// => reshape and transpose back
export class WebGlProgramSum2D {
    webgl: WebGL2Driver;
    x: WebGLTensor;

    inValueType: 'int'|'float';
    outValueType: 'int' | 'float';

    constructor(webgl:WebGL2Driver, x: WebGLTensor) {
        this.webgl = webgl;
        this.x = x;
        ASSERT(x.shape.length == 2 && x._array.isOriginalCore(), "Please preprocessing input for reduce sum");
        this.inValueType = (x._dtype == 'float32') ? 'float' : 'int';
        this.outValueType = this.inValueType;
    }

    generateCode() : string {
        const X = this.x;
        const [batchSize, N] = this.x.shape;
        const shapeY = [batchSize, 1];
        const Y = new WebGLTensor(new NdArray(null, shapeY), X._dtype);
        Y.calc2DTextureSize(this.webgl);

        const initVale: string = (this.outValueType == 'int') ? '0' : '0.0';

        return `#version 300 es
precision highp float;
precision highp int;

in vec2 outTex;
uniform sampler2D X;

out vec4 outColor;

${GlslCodeUtil.glslGet(X, 'X')}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(Y, 'Y', ['batchId', 'ReducedId'], 'outTex', '    ')}

    ${this.outValueType} r = ${initVale};
    for (int i = 0; i < ${N}; ++i) {
        r += getX(batchId, i);
    }
    outColor = vec4(r, 0.0, 0.0, 0.0);
}
`;
    }

    getProgram() : WebGLProgram {
        const fragShaderCode = this.generateCode();
        const prgKey = `ReduceSum_${fragShaderCode.length}_${simpleHash32(fragShaderCode)}`;
        let prg = this.webgl.getProgram(prgKey);
        if (prg == null) {
            console.log(fragShaderCode);
            prg = this.webgl.compileProgram(fragShaderCode);
            this.webgl.setProgram(prgKey, prg);
        }
        return prg;
    }

    run(): WebGLTensor {
        const prg = this.getProgram();
        
        const batchSize = this.x.shape[0];
        const shapeY = [batchSize, 1];
        const Y = (new WebGLTensor(new NdArray(null, shapeY), this.x._dtype));
        Y.PrepareGpuData(this.webgl);

        this.webgl.runProgram(
             prg,
             Y._texture,
             Y._texShape,
             [{name: 'X', tensor: this.x}],
             null
        );

        return Y;
    }
};