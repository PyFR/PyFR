// The configuration of the LU inverse: the problem, from its caller, and the
// tuning knobs of the kernels, from the policy below.
//
// The problem is the order N of the matrices, the types of the
// factorisation T (double or float), of the Jacobians ST, of the inverses
// OT and of the scale FT, the index type IX of the memory model, the tiling
// TR x TC (with PADC padded columns) of the inverses in PyFR's matrices,
// the number of dies NDIES the blocks are dealt across, and whether a
// launch has few matrices (FEW: at most three per CU), where more blocks
// per CU gain nothing.
//
// The knobs:
//   K1W   waves per block of the factorisation (4 or 8)
//   S     strips of 16 columns per pass of the factorisation (1 or 2)
//   K1MB  launch bounds of the factorisation: minimum waves per SIMD
//   IOL   invert the diagonal blocks out of line (0 or 1)
//   WPB   waves per block of the solves (2 or 4), sharing their loads
//   K2MB  launch bounds of the solves: minimum waves per SIMD
#pragma once

namespace luinv
{

struct knobs
{
    int k1w, s, k1mb, iol, wpb, k2mb;
};

// The knobs measured best on MI325X (gfx942) for NB strips of 16 columns,
// the factorisation in fp32 or fp64.
//
// The factorisation: fp32 tiles take half the registers of fp64, so fp32
// runs more blocks per CU and keeps four waves to larger matrices; the
// irregular bounds follow where each shape starts to spill.  Eight waves
// and two strips per pass cut the latency of each big matrix instead.
// Inverting the diagonal blocks out of line frees registers for more
// blocks, except in fp64 above 16 strips where the updates take nearly
// them all, and in fp64 it loses to the costs of more blocks when a launch
// has few matrices.  Never with eight waves: at n=625 the factorisation
// then goes wrong with pivoting, under HIPRTC and hipcc alike (it spills
// and calls two functions).
//
// The solves: two waves to a block for the fewest column blocks, where four
// leave many idle, and one wave per SIMD above 20 strips, where their
// registers spill otherwise.
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

    // The matrices padded to whole strips, and the workspace of each: its
    // strips, then the inverses of the diagonal blocks of U
    static constexpr int NP = (N + 15)/16*16, NB = NP/16;
    static constexpr long long WSZ = (long long) NP*(NP + 16);

    // The knobs
    static constexpr knobs P = policy(NB, sizeof(T) == 4, FEW);

    static constexpr int K1W = P.k1w, S = P.s, K1MB = P.k1mb;
    static constexpr bool IOL = P.iol;
    static constexpr int WPB = P.wpb, K2MB = P.k2mb;

    // Threads per block of the build, the factorisation and the solves
    static constexpr int BUILD_THREADS = 256;
    static constexpr int FACTOR_THREADS = 64*K1W;
    static constexpr int SOLVE_THREADS = 64*WPB;

    static_assert(__is_same(T, double) || __is_same(T, float),
                  "the factorisation is in fp64 or fp32");
    static_assert(K1W == 4 || K1W == 8, "four or eight waves");
    static_assert(S >= 1, "at least one strip per pass");
};

}
