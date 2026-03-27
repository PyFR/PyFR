<%inherit file='base'/>
<%namespace file='pyfr.backends.base.kernels.packing' name='pkg'/>

kernel void
pack_view(constant ixdtype_t& n, constant ixdtype_t& nrv,
          constant ixdtype_t& ncv,
          device const fpdtype_t* v,
          device const ixdtype_t* vix,
          device const ixdtype_t* vrstri,
          device fpdtype_t* pmat,
          uint i [[thread_position_in_grid]])
{
${pkg.pack_view_body()}
}

kernel void
unpack_view(constant ixdtype_t& n, constant ixdtype_t& nrv,
            constant ixdtype_t& ncv,
            device fpdtype_t* v,
            device const ixdtype_t* vix,
            device const ixdtype_t* vrstri,
            device const fpdtype_t* pmat,
            uint i [[thread_position_in_grid]])
{
${pkg.unpack_view_body()}
}
