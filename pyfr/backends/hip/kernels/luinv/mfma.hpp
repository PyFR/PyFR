// Operand layouts of the 16 x 16 x 4 MFMA instructions of gfx94x in fp64 and
// fp32 (v_mfma_f64_16x16x4_f64 and v_mfma_f32_16x16x4_f32).  A wave computes
// D = A.B + C, with A 16x4, B 4x16 and C, D 16x16.
//
// Lane l holds A[l % 16][l / 16] and B[l / 16][l % 16].  Of the four
// accumulator values v = 0..3, it holds
//   fp64:  C[l / 16 + 4*v][l % 16]
//   fp32:  C[4*(l / 16) + v][l % 16]
//
// So accumulator value v of a tile Z is row mfma_crow(l, v) of Z.  A product
// M.Z can use it as the B operand of slice v, if the A operand of that slice
// holds the matching columns of M, M[l % 16][mfma_crow(l, v)].  A solved
// tile can then feed the next product directly from its registers.
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

// Row of the A operand element held by the lane
__device__ __forceinline__ int mfma_lrow(int lane) { return lane % 16; }

// Row of accumulator value v held by the lane
template<typename T>
__device__ __forceinline__ int mfma_crow(int lane, int v)
{
    if constexpr (sizeof(T) == 8)
        return lane / 16 + 4*v;
    else
        return 4*(lane / 16) + v;
}

}
