<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
reduction(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim,
          fpdtype_t *__restrict__ reduced,
% for v in vvars:
          fpdtype_t *__restrict__ ${v}${',' if not loop.last or svars else ')'}
% endfor
% for s in svars:
          fpdtype_t ${s}${')' if loop.last else ','}
% endfor
{
% if pvars:
    #define VARIDX blockIdx.y
% for i, name in enumerate(pvars):
    const fpdtype_t *_pv_${name} = _pv + ${i*ncola};
% endfor
% endif
    int tid = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    int nwarps = blockDim.x / warpSize;
    ixdtype_t i = ixdtype_t(blockIdx.x)*blockDim.x + threadIdx.x;

    __shared__ fpdtype_t sdata[${nexprs}][32];
    fpdtype_t acc[${nexprs}] = ${pyfr.array(str(init_val), i=nexprs)};

    if (i < ncolb)
    {
        for (ixdtype_t j = 0; j < nrow; j++)
        {
            ixdtype_t idx = j*ldim + SOA_IX(i, blockIdx.y, gridDim.y);
            % for i, e in enumerate(exprs):
            % if rop == 'max':
            acc[${i}] = max(acc[${i}], ${e});
            % else:
            acc[${i}] += ${e};
            % endif
            % endfor
        }
    }

    // Reduce within each warp
    for (int off = warpSize / 2; off > 0; off >>= 1)
    {
        for (int i = 0; i < ${nexprs}; i++)
        {
        % if rop == 'max':
            acc[i] = max(__shfl_down_sync(0xFFFFFFFFU, acc[i], off), acc[i]);
        % else:
            acc[i] += __shfl_down_sync(0xFFFFFFFFU, acc[i], off);
        % endif
        }
    }

    // Have the first thread in each warp write out to shared memory
    if (tid == 0)
    {
        for (int i = 0; i < ${nexprs}; i++)
            sdata[i][wid] = acc[i];
    }

    __syncthreads();

    // Final phase: assign each warp to expression(s)
    if (wid < ${nexprs})
    {
        for (int e = wid; e < ${nexprs}; e += nwarps)
        {
            fpdtype_t acc_e = (tid < nwarps) ? sdata[e][tid] : ${init_val};

            // Final warp shuffle
            for (int off = warpSize / 2; off > 0; off >>= 1)
            {
            % if rop == 'max':
                acc_e = max(__shfl_down_sync(0xFFFFFFFFU, acc_e, off), acc_e);
            % else:
                acc_e += __shfl_down_sync(0xFFFFFFFFU, acc_e, off);
            % endif
            }

            // Atomic update to global result
            if (tid == 0)
            {
            % if rop == 'max':
                atomic_max_fpdtype(&reduced[e*gridDim.y + blockIdx.y], acc_e);
            % else:
                atomicAdd(&reduced[e*gridDim.y + blockIdx.y], acc_e);
            % endif
            }
        }
    }
% if pvars:
    #undef VARIDX
% endif
}
