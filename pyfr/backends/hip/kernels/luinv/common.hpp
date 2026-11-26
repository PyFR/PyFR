// What the hand written kernels share with the PyFR kernels: the fixed
// width integers HIPRTC lacks, and the numbering of the blocks.
#pragma once

#if defined(__HIPCC_RTC__)
typedef short int16_t;
typedef long long int64_t;
#endif

namespace luinv
{

// The number of the block as PYFR_BLOCK_ID gives it: each of the NDIES dies
// (XCDs) the blocks are dealt across round robin takes a contiguous range,
// so neighbouring blocks share the L2 of their die
template<int NDIES, typename IX>
__device__ __forceinline__ IX block_id()
{
    return IX((blockIdx.x % NDIES)*(gridDim.x / NDIES)
              + min(blockIdx.x % NDIES, gridDim.x % NDIES)
              + blockIdx.x / NDIES);
}

}
