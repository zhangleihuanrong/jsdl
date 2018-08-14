require('@tensorflow/tfjs-node');

const fs = require('fs');

const imageLoader = require('./src/image-loader');
const OnnxModel = require('./src/onnx-model');

const inputModelFile = process.argv[2] || '../sample-models/resnet50/model.onnx';
console.log("  ==== Using onnx model file:", inputModelFile);

const NumpyLoader = (function () {
    function asciiDecode(buf) {
        return String.fromCharCode.apply(null, new Uint8Array(buf));
    }

    function readUint16LE(buffer) {
        var view = new DataView(buffer);
        var val = view.getUint8(0);
        val |= view.getUint8(1) << 8;
        return val;
    }

    function fromArrayBuffer(buf) {
      // Check the magic number
      var magic = asciiDecode(buf.slice(0,6));
      if (magic.slice(1,6) != 'NUMPY') {
          throw new Error('unknown file type');
      }

      var version = new Uint8Array(buf.slice(6,8)),
          headerLength = readUint16LE(buf.slice(8,10)),
          headerStr = asciiDecode(buf.slice(10, 10+headerLength));
          offsetBytes = 10 + headerLength;
          //rest = buf.slice(10+headerLength);  XXX -- This makes a copy!!! https://www.khronos.org/registry/typedarray/specs/latest/#5

      // Hacky conversion of dict literal string to JS Object
      eval("var info = " + headerStr.toLowerCase().replace('(','[').replace('),',']'));
    
      // Intepret the bytes according to the specified dtype
      var data;
      if (info.descr === "|u1") {
          data = new Uint8Array(buf, offsetBytes);
      } else if (info.descr === "|i1") {
          data = new Int8Array(buf, offsetBytes);
      } else if (info.descr === "<u2") {
          data = new Uint16Array(buf, offsetBytes);
      } else if (info.descr === "<i2") {
          data = new Int16Array(buf, offsetBytes);
      } else if (info.descr === "<u4") {
          data = new Uint32Array(buf, offsetBytes);
      } else if (info.descr === "<i4") {
          data = new Int32Array(buf, offsetBytes);
      } else if (info.descr === "<f4") {
          data = new Float32Array(buf, offsetBytes);
      } else if (info.descr === "<f8") {
          data = new Float64Array(buf, offsetBytes);
      } else {
          throw new Error('unknown numeric dtype')
      }

      return {
          shape: info.shape,
          fortran_order: info.fortran_order,
          data: data
      };
    }

    function open(file, callback) {
        const b = fs.readFileSync(file);
        const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
        const ndarray = fromArrayBuffer(ab);
        callback(ndarray);
    }

    function ajax(url, callback) {
        var xhr = new XMLHttpRequest();
        xhr.onload = function(e) {
            var buf = xhr.response; // not responseText
            var ndarray = fromArrayBuffer(buf);
            callback(ndarray);
        };
        xhr.open("GET", url, true);
        xhr.responseType = "arraybuffer";
        xhr.send(null);
    }

    return {
        open: open,
        ajax: ajax
    };
})();

let inputPromise = null;

const imageName = process.argv[3] || 'test';
if (imageName.startsWith('test')) {
    const imageFilePath = process.argv[4];
    inputPromise = new Promise((resolve, reject) => {
        NumpyLoader.open(imageFilePath, (ndarray) => {
            resolve(ndarray.data);
        });
    });
}
else {
    const imageUrl = imageLoader.sampleImageUrls.find(item => item.text === imageName).value;
    console.log(`  ==== Loading image [[[${imageName}]]] from: ${imageUrl}...`);
    inputPromise = imageLoader.loadImage(imageUrl);
}


if (imageName.startsWith('test')) {
    const resultFilePath = process.argv[4].replace('inputs.npy', 'outputs.npy');
    const validatePromise = new Promise((resolve, reject) => {
        NumpyLoader.open(resultFilePath, (ndarray) => {
            resolve(ndarray.data);
        });
    });

    validatePromise.then(stdProbs => {
        const stdTop5 = imageLoader.imagenetClassesTopK(stdProbs, 5);
        console.log("===============Gold Result Top 5=================");
        stdTop5.map(topMatch => console.log(topMatch));
        console.log("===============End of Gold Result Top 5=================");
    });
}

const model = OnnxModel.readModeFile(inputModelFile);

//OnnxModel.debugShowModel(model);

inputPromise.then(matrix => {
    const results = OnnxModel.predict(model, matrix);
    const probsTensor = results[0];
    probsTensor.data().then(probs => {
        const top5 = imageLoader.imagenetClassesTopK(probs, 5);
        top5.map(topMatch => console.log(topMatch));
    });
});

let i = 100;


