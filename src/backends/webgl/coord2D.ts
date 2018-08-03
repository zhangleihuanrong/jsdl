import { WebGLTensor } from '../backend_webgl';


// Generate glsl code for coordinate mapping, get texture value for Tensor
export class CoordinateMapping {


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

        const glslTexValueType = (x._dtype == 'float32') ? 'float' : (x._dtype == 'int32') ? 'int' : 'uint';
        const codes: string[] = [`${glslTexValueType} ${functionPrefix}${name}(`];
        for (let i = 0; i < nda.shape.length; ++i) {
            codes.push((i > 0) ? `, int i${i}` : `int i${i}`);
        }
        codes.push(') {\n');

        // handle repeat/padding/gather logic if needed
        for (let i = 0; i < x.shape.length; ++i) {
            if (nda.gather) {
                if (nda.repeat && nda.repeat[i] > 1) {
                    const repWide = (nda.gather[i].length > 0) ? nda.gather[i].length : nda.coreShape[i];
                    codes.push(`  i${i} = i${i} % ${repWide};\n`);
                }
                if (nda.gather[i].length > 0) {
                    const gatherWide = nda.gather[i].length;
                    codes.push(`  const int gather${i}[${gatherWide}] = int[${gatherWide}](${nda.gather[i]});\n`);
                    codes.push(`  i${i} = gather${i}[i${i}];\n`);
                }
            }
            if (nda.padding) {
                const padTheAxis = (nda.padding[i] && (nda.padding[i][0] > 0 || nda.padding[i][1] > 0));
                if (nda.repeat && nda.repeat[i] > 1) {
                    const repWide = nda.coreShape[i] + ((padTheAxis) ? (nda.padding[i][0] + nda.padding[i][1]) : 0);
                    codes.push(`  i${i} = i${i} % ${repWide};\n`);
                }
                if (padTheAxis) {
                    const rightBoundary = nda.padding[i][0] + nda.coreShape[i];
                    // accessing the padding value
                    codes.push(`  if (i${i} < ${nda.padding[i][0]} || i${i} >= ${rightBoundary}) { return float(${nda.paddingValue}); } \n`);
                    codes.push(`  i${i} = i${i} - ${nda.padding[i][0]};\n`);
                }
            }
        }

        codes.push(`  int coreIndex = ${nda.coreOffset}`);
        for (let i = 0; i < nda.coreShape.length; ++i) {
            codes.push(` + i${i} * ${nda.coreStride[i]}`);
        }
        codes.push('; \n')

        codes.push(`
  int texY = int(coreIndex / ${x._texShape[0]}); // texW
  int texX = coreIndex - (texY * ${x._texShape[0]});
  return texelFetch(${name}, ivec2(texX, texY), 0).r;
`);
        codes.push('}\n');

        return codes.join('');
    }

    // this tensor is for render output, so it only contains shape/stride
    static snippetLogicFormST(x: WebGLTensor, name: string, indexPrefix = 'i', stPrefix: string = 'outTex'): string {

//         if (x._array.shape.length == 2 && x._array.shape[0] == x._texShape[1] && x._array.shape[1] == x._texShape[0]) {
//             return `
//   int ${indexPrefix}${name}_1  = int(float(${x._array.shape[1]}) * ${stPrefix}.s); //${name}._texW = ${x._array.shape[1]}
//   int ${indexPrefix}${name}_0  = int(float(${x._array.shape[0]}) * ${stPrefix}.t); //${name}._texH = ${x._array.shape[0]}
// `;
//         }
        
        const nda = x._array;
        const codes: string[] = [];
        codes.push(`
  int ${indexPrefix}${name}_x  = int(float(${x._array.shape[1]}) * ${stPrefix}.s); //${name}._texW = ${x._array.shape[1]}
  int ${indexPrefix}${name}_y  = int(float(${x._array.shape[0]}) * ${stPrefix}.t); //${name}._texH = ${x._array.shape[0]}
  int indexOf${name} = ${indexPrefix}${name}_y * ${x._array.shape[1]} + ${indexPrefix}${name}_x;

`);

      for (let i = 0; i < nda.coreShape.length; ++i) {
          const stride = nda.coreStride[i];
          codes.push(`  int ${indexPrefix}${name}_${i} =  int(indexOf${name} / ${stride});\n`);
          codes.push(`  indexOf${name} -= ${indexPrefix}${name}_${i} * ${stride};\n`);
      }
      codes.push('\n');
      return codes.join('');
    }
}