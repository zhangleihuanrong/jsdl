PBJS=../node_modules/protobufjs/bin/pbjs

all : ../src/onnx.js


../src/onnx.js : onnx-ml.proto
	$(PBJS) -t static-module -w closure -r onnx -o ../src/onnx.js onnx-ml.proto

