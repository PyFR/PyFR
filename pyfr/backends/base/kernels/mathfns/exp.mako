${qual}static inline fpdtype_t ${cname}(fpdtype_t x)
{
    // Reduce the argument to r = x - n*ln(2) with |r| <= ln(2)/2
    fpdtype_t n = rint(x*${log2e}${fsuffix});
    fpdtype_t r = fma(-n, ${ln2hi}${fsuffix}, x);
    r = fma(-n, ${ln2lo}${fsuffix}, r);

    // Approximate exp(r) by its Taylor series
    fpdtype_t p = ${poly};

    // Scale by 2^n by assembling its exponent bits directly
    union { fpdtype_t f; fp_uint_t u; } _b;
    _b.u = (fp_uint_t)((int)n + ${ebias}) << ${eshift};
    return p*_b.f;
}
