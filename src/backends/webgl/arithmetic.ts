import { WebGLTensor } from "../backend_webgl";
import { WebGL2Driver } from "./webgl2";
import { GlslCodeUtil } from './glslCodeUtil';

import { NDView as NdArray } from '../../NdView/ndview';
import { assert as ASSERT, simpleHash32 } from '../../utils/gadget';
import * as BroadcastUtil from '../../utils/broadcast';
import { DataType } from "../../types";

export type WebGLBinaryOpType = 'add' | 'sub' | 'mul' | 'div' | 'floorDiv' | 'max' | 'min' | 'mod' | 'pow' | 'squareDiff';
export type GlslDataType = 'float' | 'int';

export class WebGLBinaryOp {
    webgl: WebGL2Driver;
    opName: WebGLBinaryOpType ;
    a: WebGLTensor;
    b: WebGLTensor;
    broadcast: boolean = false;

    private static sSnippets = {
        add: 'a + b',
        sub: 'a - b',
        mul: 'a * b',
        div: 'a / b',
        floorDiv : 'ceil(a / b)',
        max: '(a > b) ? a : b',
        min: '(a < b) ? b : a',
        mod: 'mod(a, b)',
        pow: 'pow(a, b)',
        squareDiff: '(a - b) * (a - b)'
    };
    
   
    getResultTensorDataType(): DataType {
        return 'float32'; // TODO
    }

    getResultGlDataType() : GlslDataType {
        if (this.opName == 'floorDiv') return 'int';
        return 'float'; // TODO
    }

    constructor(webgl:WebGL2Driver, opName: WebGLBinaryOpType, a: WebGLTensor, b: WebGLTensor, broadcast: boolean = true) {
        this.webgl = webgl;
        this.opName = opName;
        this.a = a;
        this.b = b;
        this.broadcast = broadcast;
        if (this.broadcast) {
            ASSERT(BroadcastUtil.canBroadcastTo(this.a.shape, this.b.shape), 'Can not broadcast to A');
            const rep = BroadcastUtil.getBroadcastRepeats(this.a.shape, this.b.shape);
            this.b = this.b.tile(rep);
        }
        ASSERT(BroadcastUtil.areShapesEqual(this.a.shape, this.b.shape), "Shape not match!");
    }

    generateCode() : string {
        const A = this.a;
        const B = this.b;
        const rank = A.shape.length;

        const valueTypeA = WebGL2Driver.getGlslTextureDataType(A._dtype);
        const valueTypeB = WebGL2Driver.getGlslTextureDataType(B._dtype);

        const shapeC = A.shape;
        const C = new WebGLTensor(new NdArray(null, shapeC), this.getResultTensorDataType());
        C.calc2DTextureSize(this.webgl);

        const outGlDataType = this.getResultGlDataType();
        const castGlDataType = (valueTypeA == 'float' || valueTypeB == 'float' || outGlDataType == 'float') ? 'float' : 'int';
        const opCodeSnippet = WebGLBinaryOp.sSnippets[this.opName];

        return `#version 300 es
precision highp float;
precision highp int;
/////////////////////////////////////
//  Arithmetic_${this.opName}
/////////////////////////////////////
in vec2 outTex;
uniform sampler2D A;
uniform sampler2D B;
out vec4 outColor;

${GlslCodeUtil.glslGet(A, 'A')}

${GlslCodeUtil.glslGet(B, 'B')}

void main() {
    ${GlslCodeUtil.snippetLogicFormST(C, 'C', 'idx_', 'outTex', '    ')}

    ${castGlDataType} a = ${castGlDataType}(getA(${GlslCodeUtil.argList(rank, 'idx_')}));
    ${castGlDataType} b = ${castGlDataType}(getB(${GlslCodeUtil.argList(rank, 'idx_')}));

    ${outGlDataType} c = ${opCodeSnippet};

    outColor = vec4(c, 0.0, 0.0, 0.0);
}
`;
    }

    getProgram() : WebGLProgram {
        const fragShaderCode = this.generateCode();
        const prgKey = `Arithmetic_${this.opName}_${fragShaderCode.length}_${simpleHash32(fragShaderCode)}`;
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
        
        const C = new WebGLTensor(new NdArray(null, this.a.shape), this.getResultTensorDataType());
        C.PrepareGpuData(this.webgl);

        this.webgl.runProgram(
             prg,
             C._texture,
             C._texShape,
             [{name: 'A', tensor: this.a}, {name: 'B', tensor: this.b}],
             null
        );

        return C;
    }
};
