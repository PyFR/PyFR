<%inherit file='base'/>
<%namespace file='pyfr.backends.base.kernels.packing' name='pkg'/>

__global__ __launch_bounds__(${blocksz}) void
pack_view(ixdtype_t n, ixdtype_t nrv, ixdtype_t ncv,
          const fpdtype_t* __restrict__ v,
          const ixdtype_t* __restrict__ vix,
          const ixdtype_t* __restrict__ vrstri,
          fpdtype_t* __restrict__ pmat)
{
    ixdtype_t i = PYFR_GLOBAL_ID_X;
${pkg.pack_view_body()}
}

__global__ __launch_bounds__(${blocksz}) void
unpack_view(ixdtype_t n, ixdtype_t nrv, ixdtype_t ncv,
            fpdtype_t* __restrict__ v,
            const ixdtype_t* __restrict__ vix,
            const ixdtype_t* __restrict__ vrstri,
            const fpdtype_t* __restrict__ pmat)
{
    ixdtype_t i = PYFR_GLOBAL_ID_X;
${pkg.unpack_view_body()}
}
