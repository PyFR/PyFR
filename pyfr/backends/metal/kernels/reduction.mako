<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

kernel void
reduction(constant ixdtype_t& nrow, constant ixdtype_t& ncolb,
          device const ixdtype_t& ldim, device fpdtype_t* reduced,
  % for v in vvars:
          device const fpdtype_t* ${v},
  % endfor
  % for s in svars:
           constant fpdtype_t& ${s},
  % endfor
          ushort simdIdx [[thread_index_in_simdgroup]],
          ushort simdNum [[simdgroup_index_in_threadgroup]],
          ushort simdCnt [[simdgroups_per_threadgroup]],
          uint2 threadIdx [[thread_position_in_threadgroup]],
          uint2 blockIdx [[threadgroup_position_in_grid]])
{
% for name, values in pvars.items():
    const fpdtype_t _pv_${name}[] = ${pyfr.carray(values)};
% endfor
% if pvars:
    #define VARIDX blockIdx.y
% endif
    int tid = threadIdx.x;
    ixdtype_t i = ixdtype_t(blockIdx.x)*${blocksz} + tid;

    fpdtype_t acc[${nexprs}] = ${pyfr.array(str(init_val), i=nexprs)};
    threadgroup fpdtype_t sdata[${nexprs}][${blocksz // 8}];

    if (i < ncolb)
    {
        for (ixdtype_t j = 0; j < nrow; j++)
        {
            ixdtype_t idx = j*ldim + SOA_IX(i, blockIdx.y, ${ncola});
        % for i, e in enumerate(exprs):
          % if rop == 'max':
            acc[${i}] = max(acc[${i}], ${e});
          % else:
            acc[${i}] += ${e};
          % endif
        % endfor
        }
    }

    // Reduce over SIMD lanes
    for (int i = 0; i < ${nexprs}; i++)
        acc[i] = simd_${rop}(acc[i]);

    // Write SIMD group results to shared memory
    if (simdIdx == 0)
    {
        for (int i = 0; i < ${nexprs}; i++)
            sdata[i][simdNum] = acc[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final phase: assign each SIMD group to expression(s)
    if (simdNum < ${nexprs})
    {
        for (int e = simdNum; e < ${nexprs}; e += simdCnt)
        {
            // Each SIMD group's first thread collects and writes its assigned expression
            if (simdIdx == 0)
            {
                // Start with first SIMD group's contribution
                fpdtype_t acc_e = sdata[e][0];

                // Collect contributions from all other SIMD groups for expression e
                for (int j = 1; j < simdCnt; j++)
                {
                % if rop == 'max':
                    acc_e = max(acc_e, sdata[e][j]);
                % else:
                    acc_e += sdata[e][j];
                % endif
                }

                // Atomically update global result
                atomic_${rop}_fpdtype(&reduced[e*${ncola} + ixdtype_t(blockIdx.y)], acc_e);
            }
        }
    }
% if pvars:
    #undef VARIDX
% endif
}
