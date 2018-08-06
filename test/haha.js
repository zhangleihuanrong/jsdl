
function gemmrfloat32arfloat32brfloat32(o, a, b, A, B) {
    var od0 = o.shape[0], od1 = o.shape[1], os0 = o.stride[0], os1 = o.stride[1], oo = o.offset, od = o.data;
    var ad0 = a.shape[0], ad1 = a.shape[1], as0 = a.stride[0], as1 = a.stride[1], ao = a.offset, ad = a.data;
    var bd0 = b.shape[0], bd1 = b.shape[1], bs0 = b.stride[0], bs1 = b.stride[1], bo = b.offset, bd = b.data;
    var i, j, k;
    
    // Clear target with Zero
    var ot0 = os1, ot1 = os0 - os1 * od1, op = oo; 
    for (i = 0; i < od0; ++i) { 
        for (j = 0; j < od1; ++j) { 
            od[op] = 0; 
            op += ot0; 
        } 
        op += ot1; 
    } 
    
    for (var i0 = od0; i0 > 0;) { 
        var w0 = 32; 
        if (i0 < 32) { 
            w0 = i0; 
            i0 = 0;
        }
        else {
            i0 -= 32; 
        } 
        for (var i1 = od1; i1 > 0;) { 
            var w1 = 32; 
            if (i1 < 32) { 
                w1 = i1; 
                i1 = 0; 
            } 
            else {
                i1 -= 32; 
            } 
            
            for (var i2 = ad1; i2 > 0;) { 
                var w2 = 32; 
                if (i2 < 32) { 
                    w2 = i2; 
                    i2 = 0; 
                } 
                else { 
                    i2 -= 32; 
                } 
                
                var ot0 = os1, ot1 = os0 - os1 * w1, op = oo + i0 * os0 + i1 * os1; 
                for (i = 0; i < w0; ++i) { 
                    for (j = 0; j < w1; ++j) { 
                        var r = 0.0; 
                        var at0 = as1, ap = ao + (i0 + i) * as0 + i2 * as1; 
                        var bt0 = bs0, bp = bo + i2 * bs0 + (i1 + j) * bs1;
                         for (k = 0; k < w2; ++k) { 
                            r += ad[ap] * bd[bp]; 
                            ap += at0; bp += bt0; 
                        } 
                        od[op] = r + od[op]; 
                        op += ot0; 
                    } 
                    op += ot1; 
                } 
            } 
        } 
    } 
}
