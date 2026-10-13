${qual}static inline fpdtype_t ${cname}(fpdtype_t x)
{
    // Raise x to the integer power r
    ${tr}

    // Split t into its exponent e and a mantissa m in [1, 2)
    union { fpdtype_t f; fp_uint_t u; } _b;
    _b.f = t;
    int _e = (int)((_b.u >> ${eshift}) & ${hex(emask)}) - ${ebias};
    _b.u = (_b.u & ${hex(mmask)}) | ${hex(one_bits)};

    // Seed m^(-1/q) with a polynomial in u = m - 1.5
    fpdtype_t m = _b.f, u = m - 1.5${fsuffix};
    fpdtype_t y = ${seed};

    // Split the exponent as e = q*qf + rf and scale by 2^(-rf/q)
    fpdtype_t ef = (fpdtype_t)_e;
    fpdtype_t qf = floor(ef * ${inv_q}${fsuffix});
    fpdtype_t rf = ef - ${q}*qf;
    y *= ${scale};

    // Scale by 2^(-qf) by assembling its exponent bits directly
    _b.u = (fp_uint_t)(${ebias} - (int)qf) << ${eshift};
    y *= _b.f;

    // Refine y = t^(-1/q) with Newton iterations
% for _ in range(nn):
    { ${yq} y = y*(${q + 1} - t*yq) * ${inv_q}${fsuffix}; }
% endfor

    // Invert to obtain x^(r/q)
    return 1 / y;
}
