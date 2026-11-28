// Storage types of the Jacobians and inverses, and conversions between them.
// Rounding is as in PyFR: half is _Float16, and bf16 is the top half of an
// fp32 rounded to nearest even.
#pragma once

namespace luinv
{

struct bf16
{
    unsigned short bits;
};

template<typename To, typename From>
__device__ __forceinline__ To cvt(From v)
{
    if constexpr (__is_same(From, bf16))
        return To(__uint_as_float((unsigned) v.bits << 16));
    else if constexpr (__is_same(To, bf16))
    {
        unsigned u = __float_as_uint(float(v));
        return bf16{(unsigned short) ((u + 0x7fff + ((u >> 16) & 1)) >> 16)};
    }
    else
        return To(v);
}

}
