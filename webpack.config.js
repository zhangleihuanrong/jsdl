const path = require('path');
const webpack = require('webpack');

module.exports = {
    entry: './dist/src/index.js',
    output: {
        path: path.resolve(__dirname, 'dist', 'bundle'),
        filename: 'jsdl.js',
    },
    stats: {
        colors: true
    },
    devtool: 'source-map'
};
