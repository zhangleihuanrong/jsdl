import { WebGL2Driver, UniformProgramInfo, TextureProgramInfo } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";
import { GlslCodeUtil } from './glslCodeUtil';
import { simpleHash32 } from '../../utils/gadget';

export class WebGlBaseProgram {
    webgl: WebGL2Driver;

    // result
    y: WebGLTensor = null;
    prgTextures: TextureProgramInfo[];
    prgUniforms: UniformProgramInfo[];
    fragShaderCode: string;
    programKey: string;

    constructor(webgl:WebGL2Driver) {
        this.webgl = webgl;
        this.y = null;
        this.prgTextures = null;
        this.prgUniforms = null;
        this.fragShaderCode = null;
        this.programKey = null;
    }

    generateFragShaderHead(name: string): string {
        const lines: string[] = [`#version 300 es
precision highp float;
precision highp int;
/////////////////////////////////////////
// ${name}
/////////////////////////////////////////
in vec2 outTex;`];
        lines.push(``);

        if (this.prgTextures) {
            this.prgTextures.forEach(v => {
                lines.push(`uniform sampler2D ${v.name};`);
            });
        }
        lines.push(``);

        if (this.prgUniforms) {
            this.prgUniforms.forEach(({value, dtype, name }) => {
                if (dtype === 'float32') {
                    if (Array.isArray(value)) {
                        lines.push(`uniform float ${name}[${value.length}];`);
                    }
                    else {
                        lines.push(`uniform float ${name};`);
                    }
                } else if (dtype === 'int32' || dtype === 'bool') {
                    if (Array.isArray(value)) {
                        lines.push(`uniform int ${name}[${value.length}];`);
                    }
                    else {
                        lines.push(`uniform int ${name};`);
                    }
                }
            });
        }
        lines.push(``);

        if (this.prgTextures) {
            this.prgTextures.forEach(v => {
                lines.push(`${GlslCodeUtil.glslGet(v.tensor, v.name)}`);
                lines.push(``);
            });
        }

        lines.push(`out vec4 outColor;`);
        lines.push(``);

        return lines.join('\n');
    }

    prepareOutput() {
        throw new Error('Virtual prepareOutput called');
    }

    generateFragShaderCode(): string {
        throw new Error('Virtual prepareOutput called');
    }

    // could be overrided in derived class if code key could be static
    getProgramKey() {
        if (this.programKey === null) {
            const fsCode = this.getFragShaderCode();
            this.programKey = `Convolution_${fsCode.length}_${simpleHash32(fsCode)}`;
        }
        return this.programKey;
    }

    getFragShaderCode(): string {
        if (this.fragShaderCode === null) {
            this.fragShaderCode = this.generateFragShaderCode();
        }
        return this.fragShaderCode;
    }

    run() {
        // check parameters, make output texture, set uniforms and textures needed to run program
        this.prepareOutput();
        
        const prgKey = this.getProgramKey();
        let prg = this.webgl.getProgram(prgKey);
        if (prg == null) {
            const fsCode = this.getFragShaderCode();

            const startCompile = new Date().getTime();
            prg = this.webgl.compileProgram(fsCode);
            let msCompile = (new Date()).getTime() - startCompile;
            console.log(`>>>>>>>>Compile glsl program cost ${msCompile}ms<<<<<<<<`);

            this.webgl.setProgram(prgKey, prg);
        }

        this.webgl.runProgram(prg, this.y._texture, this.y._texShape, this.prgTextures, this.prgUniforms);
        return this.y;
    }
}
