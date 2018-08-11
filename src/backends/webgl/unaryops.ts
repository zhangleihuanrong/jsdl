import { WebGLTensor } from "../backend_webgl";
import { WebGL2Driver, UniformProgramInfo } from "./webgl2";
import { GlslCodeUtil } from './glslCodeUtil';

import { NDView as NdArray } from '../../NdView/ndview';
import { assert as ASSERT } from '../../utils/gadget';
import { DataType } from "../../types";

export type WebGlUnaryOpType = 'exp' | 'neg' | 'relu';

export class WebGlProgramUnaryOp {
    webgl: WebGL2Driver;
    opName: WebGlUnaryOpType; // TODO: enum it
    x: WebGLTensor;

    private static resultTypeMap = {
        exp:  {float: 'float' , int: 'float'},
    }

    private static snippets = {
        relu: {float32: 'return (v > 0.0) ? v : 0.0;', int32 : 'return (v > 0) ? v : 0;'},
        neg:  {float32: 'return -v;', int32 : 'return -v;'},
        exp:  {float32: `return exp(v);`, int32: `return exp(float(v));`},
    }

    constructor(webgl:WebGL2Driver, opName: WebGlUnaryOpType, x: WebGLTensor) {
        this.webgl = webgl;
        this.opName = opName;
        this.x = x;
    }

    run(): WebGLTensor {
        const program = this.getProgram(this.x, this.opName);

        const dtype : DataType = (this.x._dtype == 'bool') ? 'int32' : this.x._dtype;
        const valueType : 'int'|'float' = (dtype == 'int32') ? 'int' : 'float';
        const resultType = WebGlProgramUnaryOp.getResultType(this.opName, valueType);

        const Y = new WebGLTensor(new NdArray(null, this.x.shape), (resultType == 'int') ? 'int32':'float32');
        Y.PrepareGpuData(this.webgl);

        const uniforms : UniformProgramInfo[] = [
            { name: 'XShape', dtype : 'int32', value: this.x._array.coreShape},
            { name: 'XStride', dtype : 'int32', value: this.x._array.coreStride},
            { name: 'XOffsetTexWTexY', dtype : 'int32', value: [this.x._array.coreOffset, this.x._texShape[0], this.x._texShape[1]]},
            { name: 'YShape', dtype : 'int32', value: Y._array.coreShape},
            { name: 'YStride', dtype : 'int32', value: Y._array.coreStride},
            { name: 'YOffsetTexWTexY', dtype : 'int32', value: [Y._array.coreOffset, Y._texShape[0], Y._texShape[1]]},
        ];

        this.webgl.runProgram(
            program,
            Y._texture,
            Y._texShape,
            [{ name: 'X', tensor: this.x}],
            uniforms);

        return this.genCodeAndRun(this.x, this.opName);
    }
    
    static getResultType(opName: string, valueType: 'int'|'float') : 'int' | 'float' {
        if (WebGlProgramUnaryOp.resultTypeMap.hasOwnProperty(opName)) {
            return WebGlProgramUnaryOp.resultTypeMap[opName][valueType];
        }
        return valueType;
    }


    // generate program based on X's rank and opName
    // cache it with name: unary-${opName}-${valueType}-${rank}
    getProgram(X: WebGLTensor, opName: string): WebGLProgram {
        const rank = X.shape.length;

        const dtype : DataType = (X._dtype == 'bool') ? 'int32' : X._dtype;
        const valueType : 'int'|'float' = (dtype == 'int32') ? 'int' : 'float';
        const resultType = WebGlProgramUnaryOp.getResultType(opName, valueType);

        const programKey = `unary-${opName}-${valueType}-${rank}`;
        let prg = this.webgl.getProgram(programKey);
        if (prg != null) return prg;

        const code = `#version 300 es
precision highp float;
precision highp int;

in vec2 outTex;
uniform sampler2D X;

uniform int XShape[${rank}];
uniform int XStride[${rank}];
uniform int XOffsetTexWTexY[3];

uniform int YShape[${rank}];
uniform int YStride[${rank}];
uniform int YOffsetTexWTexY[3];

out vec4 outColor;

${GlslCodeUtil.glslGet(X, 'X')}

${resultType} getResultOn(${valueType} v) {
    ${WebGlProgramUnaryOp.snippets[this.opName][dtype]}
}

void main() {
    ${GlslCodeUtil.generalOutputIndexFormST(rank, 'Y', 'idx_', 'outTex', '    ')}

    ${valueType} v = getX(${GlslCodeUtil.argList(rank, 'idx_')});
    v = getResultOn(v);
    outColor = vec4(v, 0.0, 0.0, 0.0);
}
`;

        console.log(code);
        const startCompile = new Date().getTime();
        prg = this.webgl.compileProgram(code);
        let msCompile = (new Date()).getTime() - startCompile;
        console.log(`>>>>>>>>Compile glsl program cost ${msCompile}ms<<<<<<<<`);
        this.webgl.setProgram(programKey, prg);
        return prg;
    }


    //pure dynamic compile and run
    genCodeAndRun(X: WebGLTensor, opName: string): WebGLTensor {
        ASSERT(X.shape.length >= 1, "Scala is not allowed in matMul");

        const shapeY = X.shape;
        const rankC = shapeY.length;
        const Y = new WebGLTensor(new NdArray(null, shapeY)).PrepareGpuData(this.webgl);

        let dtype : DataType = (X._dtype == 'bool') ? 'int32' : X._dtype;
        let valueType : 'int'|'float' = (dtype == 'int32') ? 'int' : 'float';
        let resultType = WebGlProgramUnaryOp.getResultType(opName, valueType);

        const code = `#version 300 es
precision highp float;
precision highp int;
/////////////////////////////////////
//  UnaryOp_${this.opName}
/////////////////////////////////////

in vec2 outTex;
uniform sampler2D X;
out vec4 outColor;

${GlslCodeUtil.glslGet(X, 'X')}

${resultType} getResultOn(${valueType} v) {
    ${WebGlProgramUnaryOp.snippets[this.opName][dtype]}
}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(Y, 'Y', 'idx_', 'outTex', '    ')}

    ${valueType} v = getX(${GlslCodeUtil.argList(rankC, 'idx_')});
    v = getResultOn(v);
    outColor = vec4(v, 0.0, 0.0, 0.0);
}
`;
        
        console.log(code);
        const startCompile = new Date().getTime();
        const program = this.webgl.compileProgram(code);
        let msCompile = (new Date()).getTime() - startCompile;
        console.log(`>>>>>>>>Compile glsl program cost ${msCompile}ms<<<<<<<<`);

        this.webgl.runProgram(
            program,
            Y._texture,
            Y._texShape,
            [{ name: 'X', tensor: X}], 
            null);

        this.webgl._glContext.deleteProgram(program);

        return Y;
    }

};