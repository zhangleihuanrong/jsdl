Conv:
    a) input 1X3X224X224, NCHW, inputImage.buf, uint8 array from Float32Array. Little Endian.
    b) filter 64x3x7x7,   OC IC HW, filter.buf
    c) result 1x64x112x112, NCHW    convResult.buf

