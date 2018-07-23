const path = require('path');
const webpack = require('webpack');

module.exports = {
    entry: './dist/demos/basic_tensor_samples.js',
    output: {
        path: path.resolve(__dirname, 'dist', 'web'),
        filename: 'main.bundle.js',
    },
    stats: {
        colors: true
    },
    devtool: 'source-map'
};
