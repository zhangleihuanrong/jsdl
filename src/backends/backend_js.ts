
import { Backend } from '../backend';
import { ENV } from '../environments';

import ndarray from 'ndarray';

class BackendJsCpu implements Backend {

}

const backend = new BackendJsCpu() as Backend;

ENV.registerBackend('JS_CPU', backend, 0);
