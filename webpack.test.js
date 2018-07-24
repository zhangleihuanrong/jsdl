const path = require('path');
const webpack = require('webpack');

module.exports = {
    entry: './dist/test/test_simple.js',
    output: {
        path: path.resolve(__dirname, 'dist', 'web'),
        filename: 'test.bundle.js',
    },
    stats: {
        colors: true
    },
    devtool: 'source-map'
};
