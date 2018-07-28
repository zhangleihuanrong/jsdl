'use strict'

function View3dfloat32(a, b0, b1, b2, c0, c1, c2, d) {
    this.data = a
    this.shape = [b0, b1, b2]
    this.stride = [c0, c1, c2]
    this.offset = d | 0
}

var proto = View3dfloat32.prototype
proto.dtype = 'float32'
proto.dimension = 3
Object.defineProperty(proto, 'size', {
    get: function View3dfloat32_size() {
        return this.shape[0] * this.shape[1] * this.shape[2]
    }
})
Object.defineProperty(proto, 'order', {
    get: function View3dfloat32_order() {
        var s0 = Math.abs(this.stride[0]),
            s1 = Math.abs(this.stride[1]),
            s2 = Math.abs(this.stride[2]);
        if (s0 > s1) {
            if (s1 > s2) {
                return [2, 1, 0];
            } else if (s0 > s2) {
                return [1, 2, 0];
            } else {
                return [1, 0, 2];
            }
        } else if (s0 > s2) {
            return [2, 0, 1];
        } else if (s2 > s1) {
            return [0, 1, 2];
        } else {
            return [0, 2, 1];
        }
    }
})
proto.set = function View3dfloat32_set(i0, i1, i2, v) {
    return this.data[this.offset + this.stride[0] * i0 + this.stride[1] * i1 + this.stride[2] * i2] = v
}
proto.get = function View3dfloat32_get(i0, i1, i2) {
    return this.data[this.offset + this.stride[0] * i0 + this.stride[1] * i1 + this.stride[2] * i2]
}
proto.index = function View3dfloat32_index(
    i0, i1, i2
) {
    return this.offset + this.stride[0] * i0 + this.stride[1] * i1 + this.stride[2] * i2
}
proto.hi = function View3dfloat32_hi(i0, i1, i2) {
    return new View3dfloat32(this.data, (typeof i0 !== 'number' || i0 < 0) ? this.shape[0] : i0 | 0, (typeof i1 !== 'number' || i1 < 0) ? this.shape[1] : i1 | 0, (typeof i2 !== 'number' || i2 < 0) ? this.shape[2] : i2 | 0, this.stride[0], this.stride[1], this.stride[2], this.offset)
}
proto.lo = function View3dfloat32_lo(i0, i1, i2) {
    var b = this.offset,
        d = 0,
        a0 = this.shape[0],
        a1 = this.shape[1],
        a2 = this.shape[2],
        c0 = this.stride[0],
        c1 = this.stride[1],
        c2 = this.stride[2]
    if (typeof i0 === 'number' && i0 >= 0) {
        d = i0 | 0;
        b += c0 * d;
        a0 -= d
    }
    if (typeof i1 === 'number' && i1 >= 0) {
        d = i1 | 0;
        b += c1 * d;
        a1 -= d
    }
    if (typeof i2 === 'number' && i2 >= 0) {
        d = i2 | 0;
        b += c2 * d;
        a2 -= d
    }
    return new View3dfloat32(this.data, a0, a1, a2, c0, c1, c2, b)
}
proto.step = function View3dfloat32_step(i0, i1, i2) {
    var a0 = this.shape[0],
        a1 = this.shape[1],
        a2 = this.shape[2],
        b0 = this.stride[0],
        b1 = this.stride[1],
        b2 = this.stride[2],
        c = this.offset,
        d = 0,
        ceil = Math.ceil
    if (typeof i0 === 'number') {
        d = i0 | 0;
        if (d < 0) {
            c += b0 * (a0 - 1);
            a0 = ceil(-a0 / d)
        } else {
            a0 = ceil(a0 / d)
        }
        b0 *= d
    }
    if (typeof i1 === 'number') {
        d = i1 | 0;
        if (d < 0) {
            c += b1 * (a1 - 1);
            a1 = ceil(-a1 / d)
        } else {
            a1 = ceil(a1 / d)
        }
        b1 *= d
    }
    if (typeof i2 === 'number') {
        d = i2 | 0;
        if (d < 0) {
            c += b2 * (a2 - 1);
            a2 = ceil(-a2 / d)
        } else {
            a2 = ceil(a2 / d)
        }
        b2 *= d
    }
    return new View3dfloat32(this.data, a0, a1, a2, b0, b1, b2, c)
}
proto.transpose = function View3dfloat32_transpose(i0, i1, i2) {
    i0 = (i0 === undefined ? 0 : i0 | 0);
    i1 = (i1 === undefined ? 1 : i1 | 0);
    i2 = (i2 === undefined ? 2 : i2 | 0)
    var a = this.shape,
        b = this.stride;
    return new View3dfloat32(this.data, a[i0], a[i1], a[i2], b[i0], b[i1], b[i2], this.offset)
}
proto.pick = function View3dfloat32_pick(i0, i1, i2) {
    var a = [],
        b = [],
        c = this.offset
    if (typeof i0 === 'number' && i0 >= 0) {
        c = (c + this.stride[0] * i0) | 0
    } else {
        a.push(this.shape[0]);
        b.push(this.stride[0])
    }
    if (typeof i1 === 'number' && i1 >= 0) {
        c = (c + this.stride[1] * i1) | 0
    } else {
        a.push(this.shape[1]);
        b.push(this.stride[1])
    }
    if (typeof i2 === 'number' && i2 >= 0) {
        c = (c + this.stride[2] * i2) | 0
    } else {
        a.push(this.shape[2]);
        b.push(this.stride[2])
    }
    var ctor = CTOR_LIST[a.length + 1];
    return ctor(this.data, a, b, c)
}
return function construct_View3dfloat32(data, shape, stride, offset) {
    return new View3dfloat32(data, shape[0], shape[1], shape[2], stride[0], stride[1], stride[2], offset)
}