import { DataType, TypedArray } from '../../types';
import { assert as ASSERT } from '../../utils/gadget';
//import { NDView as NdArray } from '../../NdView/ndview'

const vertexShaderSource = `#version 300 es
precision highp float;

in vec3 position;
in vec2 texcoord;
out vec2 outTex;

void main () {
  gl_Position = vec4(position, 1.0);
  outTex = texcoord;
}
`;


export class WebGL2Driver {
    _isSupported: boolean = false;
    _canvas: HTMLCanvasElement = null;
    _glContext: WebGL2RenderingContext = null;
    _vertexShader: WebGLShader = null;
    MAX_TEXTURE_SIZE: number = 0;
    MAX_TEXTURE_IMAGE_UNITS: number = 0;
    _refs = { textures: [], buffers: [] };

    constructor() {
        if (typeof window !== 'undefined') {
            this._canvas = document.createElement('canvas');
            this._glContext = this._canvas.getContext('webgl2') as WebGL2RenderingContext;
            if (this._glContext) {
                const gl = this._glContext;
                this._isSupported = true;
                gl.getExtension('EXT_color_buffer_float');
                this.MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
                this.MAX_TEXTURE_IMAGE_UNITS = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
                this.createCommonVertexShader();
            } else {
                console.error('Unable to initialize WebGL2!.');
            }
        }
    }


    createCommonVertexShader() {
        const gl = this._glContext;

        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexShaderSource);
        gl.compileShader(vertexShader);

        const success = gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS);
        if (!success) {
            console.error(gl.getShaderInfoLog(vertexShader));
            gl.deleteShader(vertexShader);
            throw new Error('Can not compile pass through vertex shader!');
        }

        this._vertexShader = vertexShader;
    }

 
    compileProgram(fragShaderSource: string): WebGLProgram {
        const gl = this._glContext;

        // create and compile fragment shader
        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragShaderSource);
        gl.compileShader(fragmentShader);

        let success = gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS);
        if (!success) {
            console.error(gl.getShaderInfoLog(fragmentShader));
            gl.deleteShader(fragmentShader);
        }

        // create program and attach compiled shaders
        const program = gl.createProgram();
        gl.attachShader(program, this._vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        success = gl.getProgramParameter(program, gl.LINK_STATUS);
        if (!success) {
            const msg = gl.getProgramInfoLog(program);
            console.error(msg);
            throw new Error(msg);
        }

        this.setupVertices(program);
        return program;
    }


    storeRef(texOrBuf: WebGLTexture | WebGLBuffer) {
        if (texOrBuf instanceof WebGLTexture) {
            this._refs.textures.push(texOrBuf);
        } else if (texOrBuf instanceof WebGLBuffer) {
            this._refs.buffers.push(texOrBuf);
        }
    }


    clearRefs() {
        const gl = this._glContext;
        this._refs.textures.forEach(texture => gl.deleteTexture(texture));
        this._refs.buffers.forEach(buffer => gl.deleteBuffer(buffer));
        this._refs = { textures: [], buffers: [] }
    }


    setupVertices(program) {
        const gl = this._glContext;

        const position = gl.getAttribLocation(program, 'position');
        const positionVertexObj = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionVertexObj);
        this.storeRef(positionVertexObj);

        const vs = new Float32Array(
            [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]);
        gl.bufferData(gl.ARRAY_BUFFER, vs, gl.STATIC_DRAW);
        gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(position);

        const texcoord = gl.getAttribLocation(program, 'texcoord');
        const texcoordVertexObj = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, texcoordVertexObj);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            gl.STATIC_DRAW);
        gl.vertexAttribPointer(texcoord, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(texcoord);
        this.storeRef(texcoordVertexObj);

        const indicesVertexObj = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indicesVertexObj);
        gl.bufferData(
            gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]),
            gl.STATIC_DRAW);
        this.storeRef(indicesVertexObj);
    }

    create2DGLTexture(data: TypedArray, texShape: [number, number], dtype: DataType): WebGLTexture {
        const gl = this._glContext;
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        const W = texShape[0];
        const H = texShape[1];
        ASSERT(W <= this.MAX_TEXTURE_SIZE && H <= this.MAX_TEXTURE_SIZE, "Texture is too big");
        const level = 0;
        const internalFormat = (dtype == 'int32') ? gl.R32I : ((dtype == 'bool') ? gl.R8UI : gl.R32F);
        const border = 0;
        const format = gl.RED;
        const pixelDataType = (dtype == 'int32') ? gl.INT : ((dtype == 'bool') ? gl.UNSIGNED_BYTE : gl.FLOAT);
        if (data != null && data.length < W * H) {
            const arrayType = (dtype == 'int32') ? 'Int32Array' : ((dtype == 'bool') ? 'Uint8Array' : 'Float32Array');
            const sz = W * H;
            const extData: TypedArray = eval(`new ${arrayType}(${sz})`);
            for (let i = 0; i < sz; ++i) {
                extData[i] = (i < data.length) ? data[i] : 0;
            }
            data = extData;
        }
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, W, H, border, format, pixelDataType, data);

        // clamp to edge
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // no interpolation
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        return texture;
    }

    // return W,H for texture
    calc2DTextureSizeForFlatLen(flatLen: number): [number, number] {
        const W = Math.ceil(Math.sqrt(flatLen));
        const H = Math.ceil(flatLen / W);
        return [W, H];
    }

    // return W,H for texture
    calc2DTextureSizeForShape(shape: number[]): [number, number] {
        if (shape.length == 2 && shape.every(v => v <= this.MAX_TEXTURE_SIZE)) {
            return [shape[1], shape[0]];
        }
        else {
            const flatLen = shape.reduce((m, v) => m * v, 1);
            return this.calc2DTextureSizeForFlatLen(flatLen);
        }
    }

    bindUniforms(program: WebGLProgram, uniforms: { value: number | number[], type: DataType, name: string }[]) {
        if (uniforms) {
            const gl = this._glContext;
            uniforms.forEach(({ value, type, name }) => {
                const loc = gl.getUniformLocation(program, name);
                if (type === 'float32') {
                    if (Array.isArray(value)) {
                        eval(`gl.uniform${value.length}fv(loc, value)`);
                    }
                    else {
                        gl.uniform1f(loc, value as number);
                    }
                } else if (type === 'int32' || type === 'bool') {
                    if (Array.isArray(value)) {
                        eval(`gl.uniform${value.length}fi(loc, value)`);
                    }
                    else {
                        gl.uniform1i(loc, value as number);
                    }
                }
            });
        }
    }


    bindInputTextures(program: WebGLProgram, inTextures: { name: string, texture: WebGLTexture }[]) {
        const gl = this._glContext;

        inTextures.forEach(({ name, texture }, i) => {
            gl.activeTexture(gl.TEXTURE0 + i);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.uniform1i(gl.getUniformLocation(program, name), i);
        }
        );
    }


    bindOutputTexture(outputTexture: WebGLTexture, shape: number[], framebuffer?: WebGLFramebuffer): WebGLFramebuffer {
        const gl = this._glContext;
        gl.viewport(0, 0, shape[0], shape[1]);
        const activeFramebuffer = framebuffer || gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, activeFramebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);
        // check if you can read from this type of texture.
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) {
            throw new Error('Can not read from this type of texture.')
        }
        return activeFramebuffer;
    }


    runProgram(program: WebGLProgram,
        outTexture: WebGLTexture,
        outTexShape: [number, number], // [W, H]
        inTextures: { name: string, texture: WebGLTexture }[],
        uniforms: { value: number | number[], type: DataType, name: string }[]) {

        const gl = this._glContext;
        gl.useProgram(program);
        this.bindUniforms(program, uniforms);
        this.bindInputTextures(program, inTextures);
        const fb = this.bindOutputTexture(outTexture, outTexShape);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fb);
    }
    

    // flatLen could be little smaller than W * H.
    DumpTexture(texture: WebGLTexture, texShape: [number, number], flatLen: number = 0, dtype: DataType = 'float32'): TypedArray {
        const gl = this._glContext;
        const W = texShape[0];
        const H = texShape[1];

        if (flatLen <= 0) flatLen = W * H;
        let ndx: TypedArray = null;
        // Create a framebuffer backed by the texture
        const framebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        try {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
            if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) {
                throw new Error('Can not read from this type of texture.')
            }
            // Read the contents of the framebuffer (data stores the pixel data)
            const len = W * H * 4;
            const arrayType = (dtype == 'int32') ? 'Int32Array' : ((dtype == 'bool') ? 'Uint8Array' : 'Float32Array');
            const pixelDataType = (dtype == 'int32') ? gl.INT : ((dtype == 'bool') ? gl.UNSIGNED_BYTE : gl.FLOAT);
            const rgbaData: TypedArray = eval(`new ${arrayType}(${len})`);
            gl.readPixels(0, 0, W, H, gl.RGBA, pixelDataType, rgbaData);
            ndx = eval(`new ${arrayType}(${flatLen})`);
            for (let i = 0; i < flatLen; ++i) {
                ndx[i] = rgbaData[i * 4];
            }
        } finally {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.deleteFramebuffer(framebuffer);
        }
        return ndx;
    }
}