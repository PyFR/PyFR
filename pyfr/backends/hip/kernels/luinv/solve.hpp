// Inverse of a matrix from its LU factorisation with MFMA, in fp64 or fp32.
//
// The factorisation gives P A = L U, with L lower and U unit upper, and the
// row order P in perm.  The inverse is A^-1 = U^-1 L^-1 P.  We solve
// L U Z = I column by column, so Z = U^-1 L^-1.  Then column k of Z is
// column perm[k] of A^-1.
//
// lu_factor (factor.hpp) packs the factor as NB x NB blocks of 16 x 16
// (NB = NP/16), one strip of 16 columns after another.  Each block is in the
// order of the MFMA A operands, so a wave loads it in one piece.  The
// inverses of the diagonal blocks of L replace the blocks themselves.  The
// inverses of the diagonal blocks of U come after all the blocks.
//
// In lu_solve_mfma each wave solves for 16 columns of Z and keeps them in
// registers as NB accumulator tiles of 16 x 16.  Block row i of a
// triangular solve is one product with the inverse of the diagonal block,
// D_ii^-1 Z_i.  Then the block rows still to be solved are updated,
// Z_j -= M_ji Z_i.  Z_i is used as the B operand directly from its
// accumulators (see mfma.hpp).  The waves of a block share the loads of the
// factor through LDS.
#pragma once

#include "mfma.hpp"
#include "types.hpp"

namespace luinv
{

// Offset in the packed factor of block (j, i) of strip i, and of the
// inverse of U_ii (after the strips)
template<int NB>
__device__ __forceinline__ int pk_block(int j, int i) { return 256*(NB*i + j); }

template<int NB>
__device__ __forceinline__ int pk_uinv(int i) { return 256*(NB*NB + i); }

// Position of element (r, c) in a packed block.  It gives the lane that
// holds the element as an A operand and the slice of the product.  Slice v
// uses the rows of Z in accumulator value v.
template<typename T>
__device__ __forceinline__ int pk_ix(int r, int c)
{
    if constexpr (sizeof(T) == 8)
        return 4*(r + 16*(c % 4)) + c / 4;
    else
        return 4*(r + 16*(c / 4)) + c % 4;
}

// The A operands held by the lane for the packed 16 x 16 block at m
template<typename T>
__device__ __forceinline__ vec4<T> load_a(const T* m, int lane)
{
    return *reinterpret_cast<const vec4<T>*>(m + 4*lane);
}

// acc +/- M . Z for the block M whose A operands are a
template<bool Neg, typename T>
__device__ __forceinline__ vec4<T> mma_a(vec4<T> a, vec4<T> z, vec4<T> acc)
{
#pragma unroll
    for (int s = 0; s < 4; s++)
        acc = mfma_16x16x4(Neg ? -a[s] : a[s], z[s], acc);

    return acc;
}

// Solve L U Z = I.  Each of the WPB waves of a block solves for 16 columns
// of Z, starting at column block cb0 + w.  A sweep goes through the factor
// one column of blocks at a time, with the inverse of the diagonal block
// first.  The blocks are staged in LDS in chunks of WPB blocks.  Each wave
// fetches one block of the next chunk while the current chunk is applied,
// so each chunk needs one barrier.
template<int NB, int WPB, typename T>
__device__ __forceinline__ void solve_columns(const T* pk, int cb0, int w,
                                              int lane, vec4<T> (&z)[NB])
{
    __shared__ vec4<T> st[2][WPB][64];

    const int cb = cb0 + w;
    const vec4<T> zero = {0, 0, 0, 0};

    // Start from the columns of the identity
#pragma unroll
    for (int j = 0; j < NB; j++)
    {
#pragma unroll
        for (int v = 0; v < 4; v++)
            z[j][v] = (j == cb && mfma_crow<T>(lane, v) == mfma_lrow(lane));
    }

    // Block m of column i of the forward or backward sweep, and the number
    // of blocks
    auto item = [&](bool fwd, int i, int m)
    {
        if (fwd)
            return pk + pk_block<NB>(i + m, i);
        else
            return pk + (m == 0 ? pk_uinv<NB>(i) : pk_block<NB>(m - 1, i));
    };
    auto count = [](bool fwd, int i) { return fwd ? NB - i : i + 1; };

    vec4<T> pf;
    auto fetch = [&](bool fwd, int i, int c)
    {
        int m = WPB*c + w;
        if (m < count(fwd, i))
            pf = load_a(item(fwd, i, m), lane);
    };

    // Apply chunk c of column i, then fetch the next chunk
    int buf = 0;
    auto chunk = [&](bool fwd, int i, int c, bool active)
    {
        if (WPB*c + w < count(fwd, i))
            st[buf][w][lane] = pf;
        __syncthreads();

        // Fetch the next chunk: from this column, the next column, or the
        // first column of the backward sweep
        if (WPB*(c + 1) < count(fwd, i))
            fetch(fwd, i, c + 1);
        else if (fwd && i + 1 < NB)
            fetch(true, i + 1, 0);
        else if (fwd)
            fetch(false, NB - 1, 0);
        else if (i > 0)
            fetch(false, i - 1, 0);

        if (active)
        {
#pragma unroll
            for (int u = 0; u < WPB; u++)
            {
                const int m = WPB*c + u;
                if (m >= count(fwd, i))
                    break;

                const vec4<T> a = st[buf][u][lane];
                if (m == 0)
                    z[i] = mma_a<false>(a, z[i], zero);
                else if (fwd)
                    z[i + m] = mma_a<true>(a, z[i], z[i + m]);
                else
                    z[m - 1] = mma_a<true>(a, z[i], z[m - 1]);
            }
        }
        buf ^= 1;
    };

    // Forward: L^-1 is lower triangular, so the block rows above cb stay
    // zero
    fetch(true, cb0, 0);
#pragma unroll
    for (int i = 0; i < NB; i++)
    {
        if (i < cb0)
            continue;

#pragma unroll
        for (int c = 0; WPB*c < NB - i; c++)
            chunk(true, i, c, i >= cb);
    }

    // Backward
#pragma unroll
    for (int i = NB - 1; i >= 0; i--)
    {
#pragma unroll
        for (int c = 0; WPB*c < i + 1; c++)
            chunk(false, i, c, true);
    }
}

// Index of element (r, c) of an inverse in PyFR's tiled matrices
template<int TR, int TC, int PADC>
__device__ __forceinline__ int tiled_ix(int r, int c)
{
    return (r / TR)*(PADC / TC)*TR*TC + (c / TC)*TR*TC + (r % TR)*TC + c % TC;
}

// One block of C::WPB waves handles WPB*16 columns of one matrix.  The last
// block of a matrix may have fewer columns.  The inverse is written as C::OT
// in tiles of C::TR x C::TC.
template<class C, typename T = typename C::T, typename OT = typename C::OT,
         typename IX = typename C::IX>
__device__ __forceinline__ void lu_solve_mfma(
    IX b, OT* __restrict__ inv_v, const IX* __restrict__ inv_vix,
    const short* __restrict__ perm_v, const IX* __restrict__ wk_v,
    const T* __restrict__ ws_v)
{
    constexpr int N = C::N, NP = C::NP, WPB = C::WPB;
    constexpr int TR = C::TR, TC = C::TC, PADC = C::PADC;
    constexpr long long WSZ = C::WSZ;
    constexpr int NB = NP/16, BPM = (NB + WPB - 1)/WPB;

    const int lane = threadIdx.x % 64, w = threadIdx.x / 64;
    const IX x = b / BPM;
    const int cb0 = int(b % BPM)*WPB, cb = cb0 + w;

    const IX wk = wk_v[x];
    const T* pk = ws_v + wk*WSZ;
    const short* perm = perm_v + wk*NP;

    vec4<T> z[NB];
    solve_columns<NB, WPB>(pk, cb0, w, lane, z);

    // Write the columns in pivoted order.  Waves past the last column only
    // helped with the loads
    if (cb >= NB)
        return;

    const int gcol = perm[16*cb + mfma_lrow(lane)];
    if (gcol < N)
    {
        OT* inv = inv_v + inv_vix[x];

#pragma unroll
        for (int j = 0; j < NB; j++)
        {
#pragma unroll
            for (int v = 0; v < 4; v++)
            {
                int r = 16*j + mfma_crow<T>(lane, v);
                if (r < N)
                    inv[tiled_ix<TR, TC, PADC>(r, gcol)] = cvt<OT>(z[j][v]);
            }
        }
    }
}

}
