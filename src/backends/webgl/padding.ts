import { WebGL2Driver } from "./webgl2";
import { WebGLTensor } from "../backend_webgl";

export class WebGlProgramPad {
    webgl: WebGL2Driver;
    x: WebGLTensor;
    paddings: [number, number][];

    constructor(webgl:WebGL2Driver, x: WebGLTensor, paddings: [number, number][]) {
        this.webgl = webgl;
        this.x = x;
        this.paddings = paddings;
    }

    run(): WebGLTensor {
        return null;
    }
};