const path = require('path');
const glob = require('glob');

module.exports = {
    entry: {
        jsdl: './src/index.ts',
        test: './test/test_MatMul.ts', //glob.sync('./test/test_conv2D.ts'),
        demo:  './samples/scratch/scratch.ts',
    },
    resolve: {
        extensions: [".tsx", ".ts", ".js"]
    },
    module: {
        rules: [
            {test: /\.tsx?$/, loader: "ts-loader" }
        ]
    },
    output: {
        path: path.resolve(__dirname, 'dist', 'bundle'),
        filename: '[name].js',
    },
    stats: {
        colors: true
    },
    node: {
        fs: "empty"
    },
    devtool: 'source-map'
};
