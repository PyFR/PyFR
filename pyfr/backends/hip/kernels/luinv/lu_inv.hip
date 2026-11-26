// The LU inverse of PyFR's block-Jacobi preconditioner on AMD GPUs with
// MFMA, in separate kernels for the build, the factorisation and the
// solves; see config.hpp for the configuration and factor.hpp and solve.hpp
// for the kernels.
//
// The caller describes the problem with macros:
//   LUINV_N        the order of the matrices
//   LUINV_FTYPE    the type of the factorisation, double or float
//   LUINV_STYPE    the type of the Jacobians: double, float, _Float16 or
//                  luinv::bf16
//   LUINV_OTYPE    the type of the inverses, likewise
//   LUINV_SCALE_T  the type of the scale
//   LUINV_IXTYPE   the index type of the memory model, int or int64_t
//   LUINV_TR, LUINV_TC, LUINV_PADC
//                  the tiles of the inverses in PyFR's matrices
//   LUINV_NDIES    the number of dies the blocks are dealt across
//   LUINV_FEW      1 for launches of at most three matrices per CU
// The launch bounds of the factorisation and the solves carry their block
// sizes, from which the caller sizes their launches.

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#include "common.hpp"
#include "config.hpp"
#include "factor.hpp"
#include "solve.hpp"

using cfg = luinv::config<
    LUINV_N, LUINV_FTYPE, LUINV_STYPE, LUINV_OTYPE, LUINV_SCALE_T,
    LUINV_IXTYPE, LUINV_TR, LUINV_TC, LUINV_PADC, LUINV_NDIES, LUINV_FEW
>;

using ix_t = cfg::IX;

// A = I - scale*src in the workspaces, a block per row of 64 matrices
extern "C" __global__ __launch_bounds__(cfg::BUILD_THREADS)
void lu_inv_build(ix_t nx, cfg::FT scale, const cfg::ST* __restrict__ src_v,
                  ix_t ldsrc, const ix_t* __restrict__ wk_v,
                  cfg::T* __restrict__ ws_v)
{
    luinv::lu_build<cfg>(luinv::block_id<cfg::NDIES, ix_t>(), nx, scale,
                         src_v, ldsrc, wk_v, ws_v);
}

// P A = L U in the workspaces, a block per matrix
extern "C" __global__ __launch_bounds__(cfg::FACTOR_THREADS, cfg::K1MB)
void lu_inv_factor(ix_t _nx, const ix_t* __restrict__ wk_v,
                   cfg::T* __restrict__ ws_v, int16_t* __restrict__ perm_v)
{
    luinv::lu_factor<cfg>(luinv::block_id<cfg::NDIES, ix_t>(), wk_v, ws_v,
                          perm_v);
}

// The inverses from the factors, a block per 16*WPB columns of a matrix
extern "C" __global__ __launch_bounds__(cfg::SOLVE_THREADS, cfg::K2MB)
void lu_inv_solve(ix_t _nx, cfg::OT* __restrict__ inv_v,
                  const ix_t* __restrict__ inv_vix,
                  const int16_t* __restrict__ perm_v,
                  const ix_t* __restrict__ wk_v,
                  const cfg::T* __restrict__ ws_v)
{
    luinv::lu_solve_mfma<cfg>(luinv::block_id<cfg::NDIES, ix_t>(), inv_v,
                              inv_vix, perm_v, wk_v, ws_v);
}
