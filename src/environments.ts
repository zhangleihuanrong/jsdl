import { TensorEngine } from "./engine";
import { Backend } from "./backend";

export class Environment {
    private backendRegistry : { [name: string ] : {backend: Backend, score: number} } = {};
    private currentBackendName: string = '';
    private currentEngine: TensorEngine = null;
    private preferedName: string = '';

    constructor() {
    }

    preferBackend(name: string) {
        this.preferedName = name;
    }

    getCurrentBackendName() : string {
        return this.currentBackendName;
    }

    registerBackend(name: string, backend: Backend, score: number = 1) {
        if (name === this.preferedName) {
            score = 10000;
        }
        if (name in this.backendRegistry) {
            console.warn(`Tensor backend '${name}' already registered.`)
        }
        else {
            this.backendRegistry[name] = { backend, score };
        }
    }

    get engine() : TensorEngine {
        if (this.currentEngine == null) {
            const name = this.getBestBackend();
            if (name) this.useBackend(name);
        }
        return this.currentEngine;
    }

    get backendName(): string {
        return this.currentBackendName;
    }

    private findBackend(name: string) : Backend {
        if (!(name in this.backendRegistry)) { return null; }
        return this.backendRegistry[name].backend;
    }

    private getBestBackend() : string {
        let bestName : string = null;
        let highestScore : number = -1.0; 
        for (let name in this.backendRegistry) {
            if (this.backendRegistry[name].score > highestScore) {
                bestName = name;
                highestScore = this.backendRegistry[name].score;
            } 
        }
        return bestName;
    }

    private useBackend(name: string) {
        console.log(`==============Using backend: ${name} ================`);
        this.currentBackendName = name;
        const backend = this.findBackend(name);
        this.currentEngine = new TensorEngine(backend);
    }
}

//declare var global : any;

function getGlobalNamespace() : { __global_TensorEnv: Environment }  {
  let ns: any;
  if (typeof (window) !== 'undefined') {
      ns = window;
  }
  else if (typeof (global) !== 'undefined') {
      ns = global;
  }
  else {
      throw new Error('Could not find global namespace!');
  }
  return ns;
}

function getOrCreateEnvironment() : Environment {
    const ns = getGlobalNamespace();
    ns.__global_TensorEnv = ns.__global_TensorEnv || new Environment();
    return ns.__global_TensorEnv;
}

export const ENV = getOrCreateEnvironment();
