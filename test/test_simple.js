const chai = require('chai');
const assert = chai.assert;
const expect = chai.expect;

describe('Simple Test in JS', function() {
    it('Good Luck should work!', function() {
        assert.equal('Good Luck!', 'Good' + ' Luck!');
    });
});
