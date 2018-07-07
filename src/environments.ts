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
        this.currentEngine = new Engine(backend);

    }

    get engine() : Engine {
        if (this.currentEngine == null) {
            this.useBackend(this.getBestBackend());
        }
        return this.currentEngine;
    }

}

function getGlobalNamespace() : { ENV: Environment }  {
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

function getEnvironment() : Environment {
    const ns = getGlobalNamespace();
    ns.ENV = ns.ENV || new Environment();
    return ns.ENV;
}

export let ENV = getEnvironment();
