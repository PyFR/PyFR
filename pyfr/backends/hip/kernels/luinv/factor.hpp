// LU factorisation with partial pivoting with MFMA, in fp64 or fp32, for
// the solves of solve.hpp.
//
// The workspace of a matrix holds it as NB = NP/16 strips of 16 columns,
// each strip NP rows of 16 entries in a row (strip-major).  lu_build fills
// it with A = I - scale*src, padded with the identity.  lu_factor then
// factorises it left-looking, a strip at a time, as P A = L U with L lower
// and U unit upper (the convention of PyFR's template), overwriting each
// strip with its 16 x 16 blocks packed for the MFMA (see solve.hpp): the
// blocks of U above the diagonal, the inverse of the diagonal block of L,
// and the blocks of L below it.  The inverses of the diagonal blocks of U
// go after the strips.
//
// For strip k the block
//   1. loads the strip in the current row order into accumulator tiles,
//   2. applies the factored strips: Z_i = L_ii^-1 Z_i for the block rows
//      i < k, each followed by Z_j -= L_ji Z_i for j > i,
//   3. stores the blocks i < k, which are now U_ik,
//   4. factorises the remaining NP - 16k rows of the strip with partial
//      pivoting, a row per lane and column, as the template does,
//   5. swaps the pivoted rows in the factored strips and the row order,
//   6. stores the factored blocks and the inverses of the diagonal blocks.
// The strips to the right are untouched until loaded in the row order of
// the time, so they take the row interchanges as they are loaded.
#pragma once

#include "config.hpp"
#include "mfma.hpp"
#include "solve.hpp"
#include "types.hpp"

namespace luinv
{

// Workspace offset of entry (r, c) of a strip-major matrix
template<int NP>
__device__ __forceinline__ int sm_ix(int r, int c)
{
    return NP*16*(c / 16) + 16*r + c % 16;
}

// Two consecutive entries of src from p, aligned to their size, as T, not
// kept in the caches
template<typename T, typename ST>
__device__ __forceinline__ vec2<T> load_pair(const ST* p)
{
    if constexpr (__is_same(ST, bf16))
    {
        auto u = __builtin_nontemporal_load(
            reinterpret_cast<const unsigned*>(p)
        );
        return {cvt<T>(bf16{(unsigned short) u}),
                cvt<T>(bf16{(unsigned short) (u >> 16)})};
    }
    else
    {
        auto v = __builtin_nontemporal_load(
            reinterpret_cast<const vec2<ST>*>(p)
        );
        return {cvt<T>(v[0]), cvt<T>(v[1])};
    }
}

// Fill the strip-major workspaces with A = I - scale*src, in T, for one row
// of the 64 matrices of an AoSoA group of src per block of four waves.  Each
// wave takes 16 of the matrices, 16 columns at a time, transposing them
// through its own LDS: it loads them two matrices a lane, so the loads of a
// column coalesce, and stores them two columns a lane, so those of a matrix
// do.  The groups of src are whole, so the loads need no guard past nx.
template<class C, typename T = typename C::T, typename ST = typename C::ST,
         typename FT = typename C::FT, typename IX = typename C::IX>
__device__ __forceinline__ void lu_build(
    IX b, IX nx, FT scale, const ST* __restrict__ src_v, IX ldsrc,
    const IX* __restrict__ wk_v, T* __restrict__ ws_v)
{
    constexpr int N = C::N, NP = C::NP, SOA = 64;
    constexpr long long WSZ = C::WSZ;
    __shared__ T stw[4][16][17];

    const IX g = b / NP;
    const int r = int(b % NP), lane = threadIdx.x % 64, w = threadIdx.x / 64;
    const T scl = T(scale);
    T (*st)[17] = stw[w];

    // Order the LDS accesses of the wave, which needs no barrier
    auto wave_sync = []()
    {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    };

    for (int c0 = 0; c0 < NP; c0 += 16)
    {
        // Entries (r, c0 + i/8) of matrices 2(i % 8) and 2(i % 8) + 1
        vec2<T> v[2];
#pragma unroll
        for (int u = 0; u < 2; u++)
        {
            int i = lane + 64*u, c = c0 + i / 8, xm = 16*w + 2*(i % 8);
            v[u] = vec2<T>{0, 0};
            if (r < N && c < N && SOA*g + xm < nx)
                v[u] = load_pair<T>(&src_v[ldsrc*r + (g*N + c)*SOA + xm]);
        }
#pragma unroll
        for (int u = 0; u < 2; u++)
        {
            int i = lane + 64*u, c = c0 + i / 8, xl = 2*(i % 8);
            st[i / 8][xl] = T(r == c) - scl*v[u][0];
            st[i / 8][xl + 1] = T(r == c) - scl*v[u][1];
        }
        wave_sync();

        // Entries (r, c0 + 2(i % 8)) and the next of matrix i/8
#pragma unroll
        for (int u = 0; u < 2; u++)
        {
            int i = lane + 64*u, xl = i / 8, c = 2*(i % 8);
            IX x = SOA*g + 16*w + xl;
            if (x < nx)
                *reinterpret_cast<vec2<T>*>(
                    &ws_v[wk_v[x]*WSZ + sm_ix<NP>(r, c0 + c)]
                ) = vec2<T>{st[c][xl], st[c + 1][xl]};
        }
        wave_sync();
    }
}

// Pivot search key of entry v of row r: the larger magnitude wins, then
// the lower row; the low 12 bits of the magnitude make way for the row
template<typename T>
__device__ __forceinline__ unsigned long long pivot_key(T v, int r)
{
    unsigned long long m = __double_as_longlong(fabs(double(v)));
    return (m & ~0xfffull) | (0xfffu - r);
}

__device__ __forceinline__ int pivot_row(unsigned long long key)
{
    return 0xfff - int(key & 0xfff);
}

// Lane value of v moved by the DPP control Ctrl, rows outside RowMask
// keeping their own
template<int Ctrl, int RowMask = 0xf>
__device__ __forceinline__ unsigned long long dpp_u64(unsigned long long v)
{
    int lo = int(v), hi = int(v >> 32);
    lo = __builtin_amdgcn_update_dpp(lo, lo, Ctrl, RowMask, 0xf, false);
    hi = __builtin_amdgcn_update_dpp(hi, hi, Ctrl, RowMask, 0xf, false);
    return (unsigned long long)(unsigned) lo | (unsigned long long) hi << 32;
}

// Largest key of the wave by DPP moves, as in a reduction to lane 63
__device__ __forceinline__ unsigned long long wave_max(unsigned long long v)
{
    auto mx = [](unsigned long long a, unsigned long long b)
    {
        return a > b ? a : b;
    };

    v = mx(v, dpp_u64<0xb1>(v));        // quad_perm [1, 0, 3, 2]
    v = mx(v, dpp_u64<0x4e>(v));        // quad_perm [2, 3, 0, 1]
    v = mx(v, dpp_u64<0x141>(v));       // row_half_mirror
    v = mx(v, dpp_u64<0x140>(v));       // row_mirror
    v = mx(v, dpp_u64<0x142, 0xa>(v));  // row_bcast:15
    v = mx(v, dpp_u64<0x143, 0xc>(v));  // row_bcast:31

    int lo = __builtin_amdgcn_readlane(int(v), 63);
    int hi = __builtin_amdgcn_readlane(int(v >> 32), 63);
    return (unsigned long long)(unsigned) lo | (unsigned long long) hi << 32;
}

// Invert a 16 x 16 diagonal block d (row major in LDS) into packed blocks,
// lanes 0-15 of the first wave given taking the columns of L^-1, with L on
// and below the diagonal of d, and those of the second the columns of
// U^-1, with U unit upper above it.  idg holds the reciprocals of the
// diagonal of d.
template<typename T>
__device__ __forceinline__ void invert_diag(const T* d, const T* idg,
                                            int lane, bool lower, T* linv,
                                            T* uinv)
{
    const int c = lane % 16;

    // Column c of the inverse by substitution a column of d at a time, each
    // solved entry updating all the rows after it at once
    T y[16];
#pragma unroll
    for (int r = 0; r < 16; r++)
        y[r] = (r == c);

    if (lower && lane < 16)
    {
#pragma unroll
        for (int k = 0; k < 16; k++)
        {
            y[k] *= idg[k];
#pragma unroll
            for (int r = k + 1; r < 16; r++)
                y[r] -= d[16*r + k]*y[k];
        }

#pragma unroll
        for (int r = 0; r < 16; r++)
            linv[pk_ix<T>(r, c)] = y[r];
    }
    else if (!lower && lane < 16)
    {
#pragma unroll
        for (int k = 15; k >= 0; k--)
        {
#pragma unroll
            for (int r = 0; r < k; r++)
                y[r] -= d[16*r + k]*y[k];
        }

#pragma unroll
        for (int r = 0; r < 16; r++)
            uinv[pk_ix<T>(r, c)] = y[r];
    }
}

// invert_diag out of line, its registers then apart from those of the
// updates: fewer at the peak of the factorisation, though the updates
// around the call may be scheduled worse
template<typename T>
__device__ __noinline__ void invert_diag_ool(const T* d, const T* idg,
                                             int lane, bool lower, T* linv,
                                             T* uinv)
{
    invert_diag(d, idg, lane, lower, linv, uinv);
}

// Make the global and LDS stores of the block visible to all of its lanes
__device__ __forceinline__ void block_sync()
{
    __threadfence_block();
    __syncthreads();
}

// Factor rows r0 = 16k to NP - 1 of the panel of strip k, which the block
// left row major in the strip, a row per lane and pass.  For each column the
// waves reduce the pivot search by DPP moves and post their best candidate
// row in LDS, so after one barrier every lane picks the pivot row and reads
// it from there.  Then apply the row interchanges to the factored strips,
// the npend strips after k waiting row major and the row order rord, store
// the blocks of L below the diagonal packed and stage the diagonal block in
// dg.  Kept out of line, as its registers are separate from those of the
// updates.
template<int NP, int W, typename T>
__device__ __noinline__ void factor_panel(
    T* a, int k, int npend, short* rord, int* ipiv, T (*cand)[W][16],
    T (*grow)[16], unsigned long long (*ckey)[W], T* dg)
{
    constexpr int NB = NP/16, NL = 64*W, NQ = (NP + NL - 1)/NL;

    const int tid = threadIdx.x, lane = tid % 64, w = tid / 64;
    const int r0 = 16*k;
    T* s = a + 256*NB*k;

    T pv[NQ][16];
#pragma unroll
    for (int q = 0; q < NQ; q++)
    {
        int r = r0 + tid + NL*q;
        if (r < NP)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                pv[q][c] = s[16*r + c];
        }
    }

#pragma unroll
    for (int j = 0; j < 16; j++)
    {
        const int gk = r0 + j, b = j % 2;

        // Find the pivot: largest magnitude in column j on or below gk, the
        // first such row on a near tie
        unsigned long long key = 0;
        int mq = -1;
#pragma unroll
        for (int q = 0; q < NQ; q++)
        {
            int r = r0 + tid + NL*q;
            unsigned long long kq = pivot_key(pv[q][j], r);
            if (r >= gk && r < NP && kq > key)
            {
                key = kq;
                mq = q;
            }
        }
        const unsigned long long mine = key;
        key = wave_max(key);

        // Post the best row of the wave as it would be as the pivot row,
        // scaled right of the diagonal, and the diagonal row
#pragma unroll
        for (int q = 0; q < NQ; q++)
        {
            if (mq == q && mine == key)
            {
                const T ipv = T(1) / pv[q][j];
#pragma unroll
                for (int c = 0; c < 16; c++)
                    cand[b][w][c] = c > j ? pv[q][c]*ipv : pv[q][c];
            }
        }
        if (tid == j)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                grow[b][c] = pv[0][c];
        }
        if (lane == 0)
            ckey[b][w] = key;
        __syncthreads();

        // Pick the best of the waves
        int wb = 0;
        key = ckey[b][0];
#pragma unroll
        for (int v = 1; v < W; v++)
        {
            if (ckey[b][v] > key)
            {
                key = ckey[b][v];
                wb = v;
            }
        }
        const int mi = pivot_row(key);
        if (tid == 0)
            ipiv[j] = mi;

        // Swap in the pivot row and eliminate the rows below
        const T* pr = cand[b][wb];
#pragma unroll
        for (int q = 0; q < NQ; q++)
        {
            int r = r0 + tid + NL*q;
            if (r == gk)
            {
#pragma unroll
                for (int c = 0; c < 16; c++)
                    pv[q][c] = pr[c];
            }
            else if (r > gk && r < NP)
            {
                if (r == mi)
                {
#pragma unroll
                    for (int c = 0; c < 16; c++)
                        pv[q][c] = grow[b][c];
                }
                T lv = pv[q][j];
#pragma unroll
                for (int c = j + 1; c < 16; c++)
                    pv[q][c] -= lv*pr[c];
            }
        }
    }
    __syncthreads();

    // Apply the interchanges to the factored strips and the row order
#pragma unroll 1
    for (int j = 0; j < 16; j++)
    {
        const int gk = r0 + j, mi = ipiv[j];
        if (mi == gk)
            continue;

        for (int e = tid; e < 16*k; e += NL)
        {
            T* sb = a + 256*NB*(e / 16);
            int c = e % 16;
            int ig = 256*(gk / 16) + pk_ix<T>(gk % 16, c);
            int im = 256*(mi / 16) + pk_ix<T>(mi % 16, c);
            T t = sb[ig];
            sb[ig] = sb[im];
            sb[im] = t;
        }
        for (int e = tid; e < 16*npend; e += NL)
        {
            T* sb = a + 256*NB*(k + 1 + e / 16) + e % 16;
            T t = sb[16*gk];
            sb[16*gk] = sb[16*mi];
            sb[16*mi] = t;
        }
        if (tid == 0)
        {
            short t = rord[gk];
            rord[gk] = rord[mi];
            rord[mi] = t;
        }
    }

    // Store the factored blocks below the diagonal, and stage the diagonal
    // block
#pragma unroll
    for (int q = 0; q < NQ; q++)
    {
        int r = r0 + tid + NL*q;
        if (r < r0 + 16)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                dg[16*(r - r0) + c] = pv[q][c];
        }
        else if (r < NP)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                s[256*(r / 16) + pk_ix<T>(r % 16, c)] = pv[q][c];
        }
    }
}

// Factorise one matrix per block of W waves, S strips per pass.
//
// The waves deal the block rows of the strips between them: wave w holds
// block rows jb + w, jb + w + W, ... as tiles 0, 1, ..., with jb advancing
// by W as the leading tiles are solved.  For each step i of the updates
// the owner of block row i has published the solved Z_i through LDS; every
// wave then subtracts L_ji Z_i from its block rows below, the owner of
// block row i + 1 first, solving and publishing it for the next step, so
// each step takes one barrier.  A pass first applies the strips factored
// before it to all its strips at once, each operand serving all of them,
// then factorises its strips in turn, applying those of the pass before.
//
// The panel is spread over all the lanes too; see factor_panel.  With
// C::IOL the inversions of the diagonal blocks are out of line.
template<class C, typename T = typename C::T, typename IX = typename C::IX>
__device__ __forceinline__ void lu_factor(IX x, const IX* __restrict__ wk_v,
                                          T* __restrict__ ws_v,
                                          short* __restrict__ perm_v)
{
    constexpr int NP = C::NP, W = C::K1W, S = C::S;
    constexpr bool IOL = C::IOL;
    constexpr long long WSZ = C::WSZ;
    constexpr int NB = NP/16, NT = (NB + W - 1)/W, NL = 64*W;

    __shared__ short rord[NP];
    __shared__ T dg[256], idg[16];
    __shared__ vec4<T> zb[2][S][64];
    __shared__ T cand[2][W][16], grow[2][16];
    __shared__ unsigned long long ckey[2][W];
    __shared__ int ipiv[16];

    const int tid = threadIdx.x, lane = tid % 64, w = tid / 64;
    const IX wk = wk_v[x];
    T* a = ws_v + wk*WSZ;
    T* uinv = a + pk_uinv<NB>(0);
    const vec4<T> zero = {0, 0, 0, 0};

    // Start from the rows in their given order
    for (int r = tid; r < NP; r += NL)
        rord[r] = r;
    __syncthreads();

    for (int k0 = 0; k0 < NB; k0 += S)
    {
        const int ns = min(S, NB - k0);

        // The nsa strips from kb held as tiles, with the updates solving
        // their block rows i0 to ke - 1
        vec4<T> z[S][NT];
        int kb, nsa, i0, ke;
        auto strip = [&](int s) { return a + 256*NB*(kb + s); };
        auto held = [&](int s) { return S == 1 || s < nsa; };

        // The operands of step i of the updates for the tiles, numbered
        // from block row jb, and of the solve of block row i + 1
        vec4<T> pa[NT], pd;
        auto fetch = [&](int i, int jb)
        {
#pragma unroll
            for (int t = 0; t < NT; t++)
            {
                int j = jb + w + W*t;
                if (j > i && j < NB)
                    pa[t] = load_a(a + pk_block<NB>(j, i), lane);
            }
            if (i + 1 < ke && w == (i + 1 - i0) % W)
                pd = load_a(a + pk_block<NB>(i + 1, i + 1), lane);
        };

        // Steps ib to ib + W - 1 of the updates, those before ke, each
        // fetching the operands of the next
        auto steps = [&](int ib)
        {
#pragma unroll
            for (int u = 0; u < W; u++)
            {
                const int i = ib + u;
                if (i >= ke)
                    break;

                vec4<T> zi[S];
#pragma unroll
                for (int s = 0; s < S; s++)
                    if (held(s))
                        zi[s] = zb[u % 2][s][lane];
                const bool next = i + 1 < ke;

                vec4<T> ca[NT], cd = pd;
#pragma unroll
                for (int t = 0; t < NT; t++)
                    ca[t] = pa[t];
                if (next)
                    fetch(i + 1, u == W - 1 ? ib + W : ib);

                // Update, solve, publish and store the next block row
                const int on = (u + 1) % W, tn = (u + 1) / W;
                if (next && w == on)
                {
#pragma unroll
                    for (int s = 0; s < S; s++)
                    {
                        if (held(s))
                        {
                            vec4<T> zn = mma_a<true>(ca[tn], zi[s], z[s][tn]);
                            zn = mma_a<false>(cd, zn, zero);
                            z[s][tn] = zn;
                            zb[(u + 1) % 2][s][lane] = zn;
#pragma unroll
                            for (int v = 0; v < 4; v++)
                                strip(s)[256*(i + 1)
                                         + pk_ix<T>(mfma_crow<T>(lane, v),
                                                 mfma_lrow(lane))] = zn[v];
                        }
                    }
                }

                // Update the other block rows below
#pragma unroll
                for (int t = 0; t < NT; t++)
                {
                    int j = ib + w + W*t;
                    if (j > i + next && j < NB)
                    {
#pragma unroll
                        for (int s = 0; s < S; s++)
                            if (held(s))
                                z[s][t] = mma_a<true>(ca[t], zi[s], z[s][t]);
                    }
                }
                __syncthreads();
            }
        };

        // Run the updates from block row i0, the tiles numbered from it,
        // returning the block row the tiles are then numbered from
        auto run = [&]()
        {
            // Solve and publish block row i0, storing it as U
            if (i0 < ke)
            {
                fetch(i0, i0);
                if (w == 0)
                {
                    const vec4<T> d = load_a(a + pk_block<NB>(i0, i0), lane);
#pragma unroll
                    for (int s = 0; s < S; s++)
                    {
                        if (held(s))
                        {
                            z[s][0] = mma_a<false>(d, z[s][0], zero);
                            zb[0][s][lane] = z[s][0];
#pragma unroll
                            for (int v = 0; v < 4; v++)
                                strip(s)[256*i0
                                         + pk_ix<T>(mfma_crow<T>(lane, v),
                                                 mfma_lrow(lane))] =
                                    z[s][0][v];
                        }
                    }
                }
            }
            __syncthreads();

            // Whole groups of steps, after which the leading tiles are
            // solved
            int jb = i0;
            for (; jb + W <= ke; jb += W)
            {
                steps(jb);

#pragma unroll
                for (int s = 0; s < S; s++)
                {
#pragma unroll
                    for (int t = 0; t < NT - 1; t++)
                        z[s][t] = z[s][t + 1];
                }
            }
            if (jb < ke)
                steps(jb);

            return jb;
        };

        // Store the block rows from ke of the strips row major
        auto store_rest = [&](int jb)
        {
#pragma unroll
            for (int s = 0; s < S; s++)
            {
#pragma unroll
                for (int t = 0; t < NT; t++)
                {
                    int j = jb + w + W*t;
                    if (held(s) && j >= ke && j < NB)
                    {
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            strip(s)[256*j + 16*mfma_crow<T>(lane, v)
                                     + mfma_lrow(lane)] = z[s][t][v];
                    }
                }
            }
            block_sync();
        };

        // Load the strips of the pass in the current row order, apply the
        // strips before it and leave the rest of their rows row major
        kb = k0, nsa = ns, i0 = 0, ke = k0;
#pragma unroll
        for (int s = 0; s < S; s++)
        {
#pragma unroll
            for (int t = 0; t < NT; t++)
            {
                int j = w + W*t;
                if (held(s) && j < NB)
                {
#pragma unroll
                    for (int v = 0; v < 4; v++)
                    {
                        int r = rord[16*j + mfma_crow<T>(lane, v)];
                        z[s][t][v] = strip(s)[16*r + mfma_lrow(lane)];
                    }
                }
            }
        }

        store_rest(run());

        for (int s = 0; s < ns; s++)
        {
            const int k = k0 + s;

            // Apply the strips of the pass before this one, which have
            // swapped its rows as they were factorised
            if (S > 1 && s > 0)
            {
                kb = k, nsa = 1, i0 = k0, ke = k;
#pragma unroll
                for (int t = 0; t < NT; t++)
                {
                    int j = k0 + w + W*t;
                    if (j < NB)
                    {
#pragma unroll
                        for (int v = 0; v < 4; v++)
                            z[0][t][v] = strip(0)[16*(16*j + mfma_crow<T>(lane, v))
                                                  + mfma_lrow(lane)];
                    }
                }
                store_rest(run());
            }

            factor_panel<NP, W, T>(a, k, ns - 1 - s, rord, ipiv, cand, grow,
                                ckey, dg);
            block_sync();

            if (tid < 16)
                idg[tid] = T(1) / dg[17*tid];
            __syncthreads();

            if (w < 2)
            {
                if constexpr (IOL)
                    invert_diag_ool(dg, idg, lane, w == 0,
                                    a + pk_block<NB>(k, k), uinv + 256*k);
                else
                    invert_diag(dg, idg, lane, w == 0,
                                a + pk_block<NB>(k, k), uinv + 256*k);
            }
            block_sync();
        }
    }

    // Leave the row order for the solves
    short* perm = perm_v + wk*NP;
    for (int r = tid; r < NP; r += NL)
        perm[r] = rord[r];
}

}
