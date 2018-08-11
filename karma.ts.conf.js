// Karma configuration
// Generated on Mon Jul 23 2018 17:07:41 GMT-0700 (Pacific Daylight Time)
module.exports = function (config) {
    config.set({

        // base path that will be used to resolve all patterns (eg. files, exclude)
        basePath: '',

        plugins: [
            require("karma-mocha"),
            require("karma-chrome-launcher"),
            require("karma-mocha-reporter"),
            require("karma-typescript"),
        ],

        // frameworks to use
        // available frameworks: https://npmjs.org/browse/keyword/karma-adapter
        frameworks: ['mocha', 'karma-typescript'],


        // list of files / patterns to load in the browser
        files: [
            'src/**/*.ts',
            'test/test_current.ts',
            { pattern: 'testdata/*.buf', watched: false, included: false, served: true, nocache: false }
        ],

        proxies: {
            "/testdata/": "/base/testdata/"
        },

        // list of files / patterns to exclude
        exclude: [],

        // preprocess matching files before serving them to the browser
        // available preprocessors: https://npmjs.org/browse/keyword/karma-preprocessor
        preprocessors: {
            'src/**/*.ts': ['karma-typescript'],
            'test/test_current.ts': ['karma-typescript']
        },

        karmaTypescriptConfig: {
            tsconfig: 'tsconfig.json',
            reports: {} // Do not produce coverage html.
        },

        // test results reporter to use
        // possible values: 'dots', 'progress'
        // available reporters: https://npmjs.org/browse/keyword/karma-reporter
        reporters: ['mocha'],


        // web server port
        port: 9876,


        // enable / disable colors in the output (reporters and logs)
        colors: true,


        // level of logging
        // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
        logLevel: config.LOG_INFO,


        // enable / disable watching file and executing tests whenever any file changes
        autoWatch: true,


        // start these browsers
        // available browser launchers: https://npmjs.org/browse/keyword/karma-launcher
        browsers: ['Chrome_Remote_Debug'],

        customLaunchers: {
            Chrome_Remote_Debug: {
                base: 'Chrome',
                flags: [
                    '--show',
                    '--remote-debugging-port=9222'
                ]
            }
        },

        // Continuous Integration mode
        // if true, Karma captures browsers, runs the tests and exits
        singleRun: false,

        // Concurrency level
        // how many browser should be started simultaneous
        concurrency: Infinity
    });
}
