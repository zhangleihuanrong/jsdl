const path = require('path');
const glob = require('glob');

module.exports = {
    entry: {
        jsdl: './src/index.ts',
        test: glob.sync('./test/test_*.ts'),
        demo:  './demos/sand.ts',
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
