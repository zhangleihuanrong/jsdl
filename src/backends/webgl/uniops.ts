import { WebGLTensor } from "../backend_webgl";
import { WebGL2Driver } from "./webgl2";
import { CoordinateMapping } from './coord2D';

import { NDView as NdArray } from '../../NdView/ndview';
import { assert as ASSERT } from '../../utils/gadget';
import { DataType } from "../../types";

export class WebGlProgramUniOp {
    webgl: WebGL2Driver;
    opName: string;
    x: WebGLTensor;

    private static snippets = {
        relu: {float32: 'return (v > 0.0) ? v : 0.0;', int32 : 'return (v > 0) ? v : 0;'},
        neg:  {float32: 'return -v;', int32 : 'return -v;'}
    }

    constructor(webgl:WebGL2Driver, opName: string, x: WebGLTensor) {
        this.webgl = webgl;
        this.opName = opName;
        this.x = x;
    }

    run(): WebGLTensor {
        return this.genCodeAndRun(this.x, this.opName);
    }

    

    genCodeAndRun(X: WebGLTensor, opName: string): WebGLTensor {
        ASSERT(X.shape.length >= 1, "Scala is not allowed in matMul");

        const shapeY = X.shape;
        const rankC = shapeY.length;
        const Y = new WebGLTensor(new NdArray(null, shapeY)).PrepareGpuData(this.webgl);

        let dtype : DataType = (X._dtype == 'bool') ? 'int32' : X._dtype;
        let valueType = (dtype == 'int32') ? 'int' : 'float';

        const code = `#version 300 es
precision highp float;
precision highp int;

in vec2 outTex;
uniform sampler2D X;
out vec4 outColor;

${CoordinateMapping.glslGet(X, 'X')}

${valueType} getResultOn(${valueType} v) {
    ${WebGlProgramUniOp.snippets[this.opName][dtype]}
}

void main() {
    ${CoordinateMapping.snippetLogicFormST(Y, 'Y', 'idx_', 'outTex', '    ')}

    ${valueType} v = getX(${CoordinateMapping.argList(rankC, 'idx_')});
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
            [{ name: 'X', texture: X._texture }], 
            null);

        this.webgl._glContext.deleteProgram(program);

        return Y;
    }

};