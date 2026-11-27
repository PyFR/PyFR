// Configuration of the LU inverse: the problem (set by the caller) and the
// tuning knobs of the kernels.
//
// The problem:
//   N      order of the matrices
//   T      type of the factorisation (double or float)
//   ST     type of the Jacobians
//   OT     type of the inverses
//   FT     type of the scale
//   IX     index type
//   TR, TC, PADC
//          tiling of the inverses in PyFR's matrices (PADC padded columns)
//   NDIES  number of dies the blocks are dealt across
//   FEW    the launch has few matrices (at most three per CU), so more
//          blocks per CU do not help
//
// The knobs:
//   K1W   waves per block of the factorisation (4 or 8)
//   S     strips of 16 columns per pass of the factorisation (1 or 2)
//   K1MB  minimum waves per SIMD (launch bounds) of the factorisation
//   IOL   invert the diagonal blocks out of line (0 or 1)
//   WPB   waves per block of the solves (2 or 4); they share their loads
//   K2MB  minimum waves per SIMD (launch bounds) of the solves
#pragma once

namespace luinv
{

struct knobs
{
    int k1w, s, k1mb, iol, wpb, k2mb;
};

// Factorisation:
// - fp32 tiles use half the registers of fp64. So fp32 runs more blocks per
//   CU and uses four waves for larger matrices. The irregular bounds follow
//   where each shape starts to spill.
// - Eight waves and two strips per pass reduce the latency of big matrices.
// - Inverting the diagonal blocks out of line frees registers for more
//   blocks. It is not used in fp64 above 16 strips, where the updates need
//   nearly all registers. In fp64 it also loses when a launch has few
//   matrices.
// - Never out of line with eight waves: at n=625 the factorisation then
//   goes wrong with pivoting, under HIPRTC and hipcc alike (it spills and
//   calls two functions).
constexpr knobs policy(int nb, bool f32, bool few)
{
    knobs k = {8, 2, 1, 0, nb <= 6 ? 2 : 4, nb <= 20 ? 2 : 1};

    if (f32)
    {
        if (nb <= 13)
            k.k1w = 4, k.s = 1, k.k1mb = nb <= 8 ? 4 : 3;
        else if (nb <= 20)
            k.k1w = 4, k.s = 2, k.k1mb = 3;
        else if (nb <= 29)
            k.k1w = 4, k.s = 1, k.k1mb = 3;
        else if (nb <= 32)
            k.k1w = 4, k.s = 2, k.k1mb = 2;

        k.iol = k.k1w == 4;
    }
    else if (nb <= 26)
    {
        k.k1w = 4, k.s = 1;
        k.k1mb = nb <= 8 && !few ? 4 : 2;
        k.iol = nb <= 16 && !few;
    }

    return k;
}

template<int N_, typename T_, typename ST_, typename OT_, typename FT_,
         typename IX_, int TR_, int TC_, int PADC_, int NDIES_,
         bool FEW = false>
struct config
{
    // The problem
    using T = T_;
    using ST = ST_;
    using OT = OT_;
    using FT = FT_;
    using IX = IX_;

    static constexpr int N = N_, TR = TR_, TC = TC_, PADC = PADC_;
    static constexpr int NDIES = NDIES_;

    // The matrices padded to whole strips, and the workspace size of each
    // matrix: its strips, then the inverses of the diagonal blocks of U
    static constexpr int NP = (N + 15)/16*16, NB = NP/16;
    static constexpr long long WSZ = (long long) NP*(NP + 16);

    // The knobs
    static constexpr knobs P = policy(NB, sizeof(T) == 4, FEW);

    static constexpr int K1W = P.k1w, S = P.s, K1MB = P.k1mb;
    static constexpr bool IOL = P.iol;
    static constexpr int WPB = P.wpb, K2MB = P.k2mb;

    // Threads per block of the build, factorisation and solve kernels
    static constexpr int BUILD_THREADS = 256;
    static constexpr int FACTOR_THREADS = 64*K1W;
    static constexpr int SOLVE_THREADS = 64*WPB;

    static_assert(__is_same(T, double) || __is_same(T, float),
                  "the factorisation is in fp64 or fp32");
    static_assert(K1W == 4 || K1W == 8, "four or eight waves");
    static_assert(S >= 1, "at least one strip per pass");
    static_assert(!(IOL && K1W == 8),
                  "out-of-line inversion with eight waves miscompiles (n=625)");
    static_assert(NP <= 4096, "pivot_key packs the row into 12 bits");
    static_assert(NDIES >= 1, "at least one die");
};

}
