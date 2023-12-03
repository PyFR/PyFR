<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

kernel void
reduction(constant ixdtype_t& nrow, constant ixdtype_t& ncolb,
          device const ixdtype_t& ldim, device fpdtype_t* reduced,
          device const fpdtype_t* rcurr, device const fpdtype_t* rold,
% if method == 'errest':
          device const fpdtype_t* rerr,
          constant fpdtype_t& atol, constant fpdtype_t& rtol,
% elif method == 'resid' and dt_type == 'matrix':
          device const fpdtype_t* dt_mat, constant fpdtype_t& dt_fac,
% elif method == 'resid':
          constant fpdtype_t& dt_fac,
% endif
          ushort simdIdx [[thread_index_in_simdgroup]],
          ushort simdNum [[simdgroup_index_in_threadgroup]],
          ushort simdCnt [[simdgroups_per_threadgroup]],
          uint2 threadIdx [[thread_position_in_threadgroup]],
          uint2 blockIdx [[threadgroup_position_in_grid]])
{
    int tid = threadIdx.x;
    ixdtype_t i = ixdtype_t(blockIdx.x)*${blocksz} + tid;
    ixdtype_t nblocks = (ncolb + ${blocksz} - 1) / ${blocksz};

    fpdtype_t r, acc = 0;
    threadgroup fpdtype_t sdata[${blocksz // 8 - 1}];

    if (i < ncolb)
    {
        for (ixdtype_t j = 0; j < nrow; j++)
        {
            ixdtype_t idx = j*ldim + SOA_IX(i, blockIdx.y, ${ncola});
        % if method == 'errest':
            r = rerr[idx]/(atol + rtol*max(fabs(rcurr[idx]), fabs(rold[idx])));
        % elif method == 'resid':
            r = (rcurr[idx] - rold[idx])/(dt_fac${'*dt_mat[idx]' if dt_type == 'matrix' else ''});
        % endif

        % if norm == 'uniform':
            acc = max(r*r, acc);
        % else:
            acc += r*r;
        % endif
        }
    }

    // Reduce over SIMD lanes
    % if norm == 'uniform':
        acc = simd_max(acc);
    % else:
        acc = simd_sum(acc);
    % endif

    if (simdNum > 0 && simdIdx == 0)
        sdata[simdNum - 1] = acc;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Have the first thread perform the final reduction
    if (tid == 0)
    {
        for (int i = 1; i < simdCnt; i++)
        {
        % if norm == 'uniform':
            acc = max(acc, sdata[i - 1]);
        % else:
            acc += sdata[i - 1];
        % endif
        }

        // Copy to global memory
        reduced[ixdtype_t(blockIdx.y)*nblocks + blockIdx.x] = acc;
    }
}
