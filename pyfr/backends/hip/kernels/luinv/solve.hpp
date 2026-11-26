// Inverse of a matrix from its LU factorisation with MFMA, in fp64 or fp32.
//
// The factorisation leaves P A = L U with L lower and U unit upper, and the
// row order P in perm.  The inverse is A^-1 = U^-1 L^-1 P, so with
// Z = U^-1 L^-1 solving L U Z = I column by column, column k of Z is column
// perm[k] of A^-1.
//
// The factor comes packed from lu_factor in factor.hpp: NB x NB blocks of
// 16 x 16 (NB = NP/16), stored a strip of 16 columns at a time, each block
// in the order of the A operands of the MFMA so a wave loads it in one
// piece; the inverses of the diagonal blocks of L are in place of them, and
// those of U come after all the blocks.
//
// lu_solve_mfma has every wave solve for 16 columns of Z, holding them in
// registers as NB accumulator tiles of 16 x 16.  Each block row i of a
// triangular solve is one product with the inverse of the diagonal block,
// D_ii^-1 Z_i, then the updates Z_j -= M_ji Z_i of the block rows still to
// be solved, with Z_i the B operand straight from its accumulators (see
// mfma.hpp).  The waves of a block share the loads of the factor through
// LDS.
#pragma once

#include "mfma.hpp"
#include "types.hpp"

namespace luinv
{

// Offsets in the packed factor of block (j, i), in strip i, and of the
// inverse of U_ii after the strips
template<int NB>
__device__ __forceinline__ int pk_block(int j, int i) { return 256*(NB*i + j); }

template<int NB>
__device__ __forceinline__ int pk_uinv(int i) { return 256*(NB*NB + i); }

// Position in a packed block of element (r, c): the lane holding it as an
// A operand and the slice of the product it is in, such that slice v takes
// the rows of Z in accumulator value v
template<typename T>
__device__ __forceinline__ int pk_ix(int r, int c)
{
    if constexpr (sizeof(T) == 8)
        return 4*(r + 16*(c % 4)) + c / 4;
    else
        return 4*(r + 16*(c / 4)) + c % 4;
}

// The A operands the lane holds of the packed 16 x 16 block at m
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

// Solve L U Z = I for the 16 columns of Z from column block cb0 + w of each
// of the WPB waves of a block.  A sweep goes a column of blocks of the
// factor at a time, the inverse of the diagonal block first, in chunks of
// WPB blocks staged in LDS; each wave fetches one block of the next chunk
// while the current one is applied, so each chunk takes one barrier.
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

    // Block m of column i of the forward or backward sweep, and their count
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

        // Fetch the next chunk: of this column, the next column, or the
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

    // Forward: L^-1 is lower triangular so the block rows above cb stay zero
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

// One block of C::WPB waves per WPB*16 columns of one matrix, the last
// block of a matrix with fewer columns than that; the inverse is written as
// C::OT in tiles of C::TR x C::TC
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

    // Write the columns out in pivoted order, with the waves of the last
    // block past the end only having helped with the loads
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
