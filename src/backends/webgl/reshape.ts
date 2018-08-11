import { NDView as NdArray } from '../../NdView/ndview';
import { WebGL2Driver } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";
import { GlslCodeUtil } from './glslCodeUtil';
import { assert as ASSERT } from '../../utils/gadget';
import { WebGlBaseProgram } from './program';

export class WebGlReshapeOp extends WebGlBaseProgram {
    x: WebGLTensor;
    newShape: number[]

    constructor(webgl:WebGL2Driver, x: WebGLTensor, newShape: number[]) {
        super(webgl);
        this.x = x;
        this.newShape = newShape;

        ASSERT(this.newShape.every(v => v > 0 || v == -1), "reshape axis len must be positive or -1");
        const numberOfNeg1 = this.newShape.reduce((n, v) => n + ((v==-1)?1:0), 0);
        ASSERT(numberOfNeg1 <= 1, "At most one -1 could be used in reshape!");
        let newSize = Math.abs(this.newShape.reduce((m, v) => m*v, 1));
        if (numberOfNeg1 == 1) {
            ASSERT(this.x._array.size % newSize === 0, "-1 can not find matching size during reshape");
            const w = Math.ceil(this.x._array.size / newSize);
            this.newShape = this.newShape.map(v => (v == -1)? w : v);
            newSize *= w;
        }
        ASSERT(newSize == this.x._array.size, "Size not matching with original!");
    }

    prepareOutput() {
        this.y = new WebGLTensor(new NdArray(null, this.newShape), this.x._dtype);
        this.y.PrepareGpuData(this.webgl);
        this.prgTextures = [{name: 'X', tensor: this.x}];
    }


    generateFragShaderCode(): string {
        const rankX = this.x.shape.length;

        const fsCode =  
`${this.generateFragShaderHead('Reshape')}

void main() {
    int outTex_x = int(float(${this.x._texShape[0]}) * outTex.s);
    int outTex_y= int(float(${this.x._texShape[1]}) * outTex.t);
    int offsetInX = outTex_y * ${this.x._texShape[0]} + outTex_x;

    ${GlslCodeUtil.pureOffset2Indices(this.x.shape, 'offsetInX', 'x_', '    ')}

    float r = getX(${GlslCodeUtil.argList(rankX, 'x_')});
    outColor = vec4(r, 0.0, 0.0, 0.0);
}
`;

    console.log(fsCode);
    return fsCode;
    }

};
