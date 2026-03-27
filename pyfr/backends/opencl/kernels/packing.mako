<%inherit file='base'/>
<%namespace file='pyfr.backends.base.kernels.packing' name='pkg'/>

__kernel void
pack_view(ixdtype_t n, ixdtype_t nrv, ixdtype_t ncv,
          __global const fpdtype_t* restrict v,
          __global const ixdtype_t* restrict vix,
          __global const ixdtype_t* restrict vrstri,
          __global fpdtype_t* restrict pmat)
{
    ixdtype_t i = get_global_id(0);
${pkg.pack_view_body()}
}

__kernel void
unpack_view(ixdtype_t n, ixdtype_t nrv, ixdtype_t ncv,
            __global fpdtype_t* restrict v,
            __global const ixdtype_t* restrict vix,
            __global const ixdtype_t* restrict vrstri,
            __global const fpdtype_t* restrict pmat)
{
    ixdtype_t i = get_global_id(0);
${pkg.unpack_view_body()}
}
