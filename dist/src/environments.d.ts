import { TensorEngine } from "./engine";
import { Backend } from "./backend";
export declare class Environment {
    backendRegistry: {
        [name: string]: {
            backend: Backend;
            score: number;
        };
    };
    currentBackendName: string;
    currentEngine: TensorEngine;
    constructor();
    registerBackend(name: string, backend: Backend, score?: number): void;
    findBackend(name: string): Backend;
    getBestBackend(): string;
    useBackend(name: string): void;
    readonly engine: TensorEngine;
}
export declare let ENV: Environment;
export declare let engine: TensorEngine;
