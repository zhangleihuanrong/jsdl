import { TensorEngine } from "./engine";
import { Backend } from "./backend";

export class Environment {
    backendRegistry : { [name: string ] : {backend: Backend, score: number} } = {};
    currentBackendName: string;
    currentEngine: TensorEngine;

    constructor() {
    }

    registerBackend(name: string, backend: Backend, score: number = 1) {
        if (name in this.backendRegistry) {
            console.warn(`Tensor backend '${name}' already registered.`)
        }
        else {
            this.backendRegistry[name] = { backend, score };
        }
    }

    findBackend(name: string) : Backend {
        if (!(name in this.backendRegistry)) { return null; }
        return this.backendRegistry[name].backend;
    }

    getBestBackend() : string {
        let bestName : string;
        let highestScore : number = -1.0; 
        for (let name in this.backendRegistry) {
            if (this.backendRegistry[name].score > highestScore) {
                bestName = name;
                highestScore = this.backendRegistry[name].score;
            } 
        }
        return bestName;
    }

    useBackend(name: string) {
        this.currentBackendName = name;
        const backend = this.findBackend(name);
        this.currentEngine = new TensorEngine(backend);

    }

    get engine() : TensorEngine {
        if (this.currentEngine == null) {
            this.useBackend(this.getBestBackend());
        }
        return this.currentEngine;
    }

}

declare var global : any;

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

export let ENV = getOrCreateEnvironment();
export let engine = ENV.engine;
