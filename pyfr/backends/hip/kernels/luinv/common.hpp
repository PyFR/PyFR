#pragma once

#if defined(__HIPCC_RTC__)
typedef short int16_t;
typedef long long int64_t;
#endif

namespace luinv
{

// Block number as PYFR_BLOCK_ID gives it.  The hardware deals blocks to the
// NDIES dies (XCDs) round robin.  This remaps them so each die gets a
// contiguous range, and neighbouring blocks share the L2 cache of their die.
template<int NDIES, typename IX>
__device__ __forceinline__ IX block_id()
{
    return IX((blockIdx.x % NDIES)*(gridDim.x / NDIES)
              + min(blockIdx.x % NDIES, gridDim.x % NDIES)
              + blockIdx.x / NDIES);
}

}
