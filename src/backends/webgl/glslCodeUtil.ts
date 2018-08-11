import { WebGLTensor } from '../backend_webgl';
import { DataType } from '../../types';


// Generate glsl code for coordinate mapping, get texture value for Tensor
export class GlslCodeUtil {

    // generate code to handle normal case, where shape/strides information are passed by attributes:
    //    uniform int ${name}Shape[rank];
    //    uniform int ${name}Shape[rank];
    //    uniform int ${name}OffsetTexWTexY[3];
    static glslGetCoreOnly(rank: number, name: string, dtype: DataType, functionPrefix = 'get'): string {
        const glslTexValueType = (dtype == 'float32') ? 'float' : (dtype == 'int32') ? 'int' : 'uint';

        const codes: string[] = [`${glslTexValueType} ${functionPrefix}${name}(`];
        for (let i = 0; i < rank; ++i) {
            codes.push((i > 0) ? `, int i${i}` : `int i${i}`);
        }
        codes.push(') {\n');

        codes.push(`  int coreIndex = ${name}OffsetTexWTexY[0]`);
        for (let i = 0; i < rank; ++i) {
            codes.push(` + i${i} * ${name}Stride[${i}]`);
        }
        codes.push('; \n')

        codes.push(`  int texY = int(coreIndex / ${name}OffsetTexWTexY[1]); // texW\n`);
        codes.push(`  int texX = coreIndex - (texY * ${name}OffsetTexWTexY[1]);\n`);
        codes.push(`  return texelFetch(${name}, ivec2(texX, texY), 0).r;\n`);

        codes.push('}');
        return codes.join('');
    }


    // this tensor is for render output, so it only contains shape/stride
    // if indexNames is string, treat it as prefix, concat it with 0, 1, ... => idx_0, idx_1, ...
    static generalOutputIndexFormST(rank: number, name: string, indexNames: string|string[] = 'idx_', stPrefix: string = 'outTex', indent: string = '    '): string {
        const codes: string[] = [];
        if (!Array.isArray(indexNames)) {
            indexNames = new Array(rank).fill(indexNames);
            indexNames = indexNames.map((v, i) => `${v}${i}`);
        }

        codes.push(`int ${stPrefix}_x = int(float(${name}OffsetTexWTexY[1]) * ${stPrefix}.s); //${name}._texW\n`);
        codes.push(`${indent}int ${stPrefix}_y = int(float(${name}OffsetTexWTexY[2]) * ${stPrefix}.t); //${name}._texH\n`);
        codes.push(`${indent}int indexOf${name} = ${stPrefix}_y * ${name}OffsetTexWTexY[1] + ${stPrefix}_x;\n`);

        for (let i = 0; i < rank; ++i) {
            codes.push(`${indent}int ${indexNames[i]} =  int(indexOf${name} / ${name}Stride[${i}]);`);
            if (i != rank - 1) {
                codes.push(`\n${indent}indexOf${name} -= (${indexNames[i]} * ${name}Stride[${i}]);\n`);
            }
        }
        return codes.join('');
    }


    // float getA(int i0, int i1...) {
    //     // int coreIdx = offset + logic0 * coreStride[0] + logic1 * coreStride[1];
    //     int coreIdx = logic0 * 5 + logic1 * 1 + 0;

    //     const int texH = 3;
    //     const int texW = 4;
    //     int texY = int(coreIdx / texW);
    //     int texX = coreIdx - (texY * texW);

    //     return texelFetch(A, ivec2(texX, texY), 0).r;
    // }
    static glslGet(x: WebGLTensor, name: string, functionPrefix = 'get'): string {
        const nda = x._array;

        const glslTexValueType = (x._dtype == 'float32') ? 'float' : 'int';
        const codes: string[] = [`${glslTexValueType} ${functionPrefix}${name}(`];
        for (let i = 0; i < nda.shape.length; ++i) {
            codes.push((i > 0) ? `, int i${i}` : `int i${i}`);
        }
        codes.push(') {\n');

        if (nda.shape.length == 2 && nda.isOriginalCore() && nda.shape[0] == x._texShape[1] && nda.shape[1] == x._texShape[0]) {
            codes.push(`  return texelFetch(${name}, ivec2(i1, i0), 0).r;\n`);
        }
        else {
            // handle repeat/padding/gather logic if needed
            for (let i = 0; i < nda.shape.length; ++i) {
                if (nda.gather) {
                    if (nda.repeat && nda.repeat[i] > 1) {
                        const repWide = (nda.gather[i].length > 0) ? nda.gather[i].length : nda.coreShape[i];
                        codes.push(`  i${i} = i${i} - (${repWide} * int(i${i}/${repWide}));\n`);
                    }
                    if (nda.gather[i].length > 0) {
                        const gatherWide = nda.gather[i].length;
                        codes.push(`  const int gather${i}[${gatherWide}] = int[${gatherWide}](${nda.gather[i]});\n`);
                        codes.push(`  i${i} = gather${i}[i${i}];\n`);
                    }
                }
                else if (nda.padding) {
                    const padTheAxis = (nda.padding[i] && (nda.padding[i][0] > 0 || nda.padding[i][1] > 0));
                    if (nda.repeat && nda.repeat[i] > 1) {
                        const repWide = nda.coreShape[i] + ((padTheAxis) ? (nda.padding[i][0] + nda.padding[i][1]) : 0);
                        codes.push(`  i${i} = i${i} - (${repWide} * int(i${i}/${repWide}));\n`);
                    }
                    if (padTheAxis) {
                        const rightBoundary = nda.padding[i][0] + nda.coreShape[i];
                        // accessing the padding value
                        codes.push(`  if (i${i} < ${nda.padding[i][0]} || i${i} >= ${rightBoundary}) { return float(${nda.paddingValue}); } \n`);
                        codes.push(`  i${i} = i${i} - ${nda.padding[i][0]};\n`);
                    }
                }
                else {
                    if (nda.repeat && nda.repeat[i] > 1) {
                        const repWide = nda.coreShape[i];
                        codes.push(`  i${i} = i${i} - (${repWide} * int(i${i}/${repWide}));\n`);
                    }
                }
            }

            codes.push(`  int coreIndex = ${nda.coreOffset}`);
            for (let i = 0; i < nda.coreShape.length; ++i) {
                codes.push(` + i${i} * ${nda.coreStride[i]}`);
            }
            codes.push('; \n')

            codes.push(`  int texY = int(coreIndex / ${x._texShape[0]}); // texW\n`);
            codes.push(`  int texX = coreIndex - (texY * ${x._texShape[0]});\n`);
            codes.push(`  return texelFetch(${name}, ivec2(texX, texY), 0).r;\n`);
        } // end of common processing
        codes.push('}');
        return codes.join('');
    }

    // this tensor is for render output, so it only contains shape/stride
    // if indexNames is string, treat it as prefix, concat it with 0, 1, ... => idx_0, idx_1, ...
    static snippetLogicFormST(x: WebGLTensor, name: string, indexNames: string|string[] = 'idx_', stPrefix: string = 'outTex', indent: string = '    '): string {
        const nda = x._array;
        const codes: string[] = [];
        if (!Array.isArray(indexNames)) {
            indexNames = new Array(x.shape.length).fill(indexNames);
            indexNames = indexNames.map((v, i) => `${v}${i}`);
        }

        if (x._array.shape.length == 2 && x._array.shape[0] == x._texShape[1] && x._array.shape[1] == x._texShape[0]) {
            codes.push(`int ${indexNames[1]}  = int(float(${x._array.shape[1]}) * ${stPrefix}.s); //${name}._texW = ${x._array.shape[1]}\n`);
            codes.push(`${indent}int ${indexNames[0]}  = int(float(${x._array.shape[0]}) * ${stPrefix}.t); //${name}._texH = ${x._array.shape[0]}`);
            return codes.join('');
        }
        
        codes.push(`int ${stPrefix}_x = int(float(${x._texShape[0]}) * ${stPrefix}.s); //${name}._texW\n`);
        codes.push(`${indent}int ${stPrefix}_y = int(float(${x._texShape[1]}) * ${stPrefix}.t); //${name}._texH\n`);
        codes.push(`${indent}int indexOf${name} = ${stPrefix}_y * ${x._texShape[0]} + ${stPrefix}_x;\n`);

        for (let i = 0; i < nda.coreShape.length; ++i) {
            const stride = nda.coreStride[i];
            codes.push(`${indent}int ${indexNames[i]} =  int(indexOf${name} / ${stride});`);
            if (i != nda.coreShape.length - 1) {
                codes.push(`\n${indent}indexOf${name} -= (${indexNames[i]} * ${stride});\n`);
            }
        }
        return codes.join('');
    }

    static argList(argCount: number, indexNames: string|string[] = 'idx_', ...replaces: [number, string][]) : string {
        if (!Array.isArray(indexNames)) {
            indexNames = new Array(argCount).fill(indexNames);
            indexNames = indexNames.map((v, i) => `${v}${i}`);
        }

        const parts = [];
        for (let i = 0; i < argCount; ++i) {
            if (i > 0) parts.push(', ');
            const pair = replaces.find((v, idx) => v[0] == i);
            parts.push( pair ? pair[1] : `${indexNames[i]}` );
        }
        return parts.join('');
    }

    static snippetGet2D(texName: string, y = 'i0', x = 'i1'): string {
        return `texelFetch(${name}, ivec2(${x}, ${y}), 0).r`;
    }


}