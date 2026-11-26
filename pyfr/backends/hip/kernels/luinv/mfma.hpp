// Operand layouts of the 16 x 16 x 4 matrix fused multiply-adds of gfx94x
// in fp64 and fp32, v_mfma_f64_16x16x4_f64 and v_mfma_f32_16x16x4_f32,
// which a wave issues as D = A.B + C with A 16x4, B 4x16 and C, D 16x16.
// Lane l of the wave holds A[l % 16][l / 16] and B[l / 16][l % 16], and of
// the four accumulator values v = 0..3
//   fp64:  C[l / 16 + 4*v][l % 16]
//   fp32:  C[4*(l / 16) + v][l % 16]
//
// Either way value v of the accumulators of a tile Z is, for lane group
// g = l / 16, row mfma_crow(l, v) of Z, so a product M.Z can take it as the
// B operand of slice v, with the A operand of that slice holding the
// matching columns of M, M[l % 16][mfma_crow(l, v)].  A solved tile thus
// feeds the next product straight from its registers.
#pragma once

namespace luinv
{

template<typename T>
using vec4 = T __attribute__((ext_vector_type(4)));

template<typename T>
using vec2 = T __attribute__((ext_vector_type(2)));

typedef vec4<double> f64x4;
typedef vec4<float> f32x4;

__device__ __forceinline__ f64x4 mfma_16x16x4(double a, double b, f64x4 c)
{
    return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ f32x4 mfma_16x16x4(float a, float b, f32x4 c)
{
    return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
}

// Row and K index of the A operand element the lane holds
__device__ __forceinline__ int mfma_lrow(int lane) { return lane % 16; }
__device__ __forceinline__ int mfma_lk(int lane) { return lane / 16; }

// Row of accumulator value v of the lane
template<typename T>
__device__ __forceinline__ int mfma_crow(int lane, int v)
{
    if constexpr (sizeof(T) == 8)
        return lane / 16 + 4*v;
    else
        return 4*(lane / 16) + v;
}

}
