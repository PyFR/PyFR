${qual}static inline fpdtype_t ${cname}(fpdtype_t x)
{
    // Split x into its exponent e and a mantissa m in [1, 2)
    union { fpdtype_t f; fp_uint_t u; } _b;
    _b.f = x;
    int _e = (int)((_b.u >> ${eshift}) & ${hex(emask)}) - ${ebias};
    _b.u = (_b.u & ${hex(mmask)}) | ${hex(one_bits)};

    // Halve m if it exceeds sqrt(2) and increment e to compensate
    fpdtype_t m = _b.f;
    int big = m > ${sqrt2}${fsuffix};
    m = big ? 0.5${fsuffix}*m : m;
    fpdtype_t e = (fpdtype_t)(_e + big);

    // Approximate log(m) = f - s*(f - r) using the series for 2*atanh(s)
    fpdtype_t f = m - 1, s = f / (2 + f), w = s*s;
    fpdtype_t r = w*(${poly});
    fpdtype_t lm = f - s*(f - r);
% if lbe != 1:

    // Convert log(m) into our base
    lm *= ${lbe}${fsuffix};
% endif

    // Add in e*log(2) to obtain log(x)
% if lb2lo:
    return fma(e, ${lb2hi}${fsuffix}, fma(e, ${lb2lo}${fsuffix}, lm));
% else:
    return fma(e, ${lb2hi}${fsuffix}, lm);
% endif
}
