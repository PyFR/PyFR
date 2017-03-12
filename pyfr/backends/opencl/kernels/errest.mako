# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__kernel void
errest(int nrow, int ncolb, int ldim, __global fpdtype_t *__restrict__ err,
       ${', '.join('__global const fpdtype_t* restrict {0}'.format(i)
                   for i in ['x', 'y', 'z'])},
       fpdtype_t atol, fpdtype_t rtol)

{
    __local fpdtype_t ${','.join('sdata{0}[{1}]'.format(i, sharesz)
                                    for i in range(ncola))};
    fpdtype_t r;
    int idx;
    int i = get_global_id(0), tid = get_local_id(0);
    int bid = get_group_id(0), sharesz = get_local_size(0);
    int gdim = get_num_groups(0);
    int lastblksize = ncolb % sharesz;

    if (i < ncolb)
    {
        // All threads load the values of the first row to local memory
    % for k in range(ncola):
        idx = SOA_IX(i, ${k}, ${ncola});
        sdata${k}[tid] = pow(x[idx]/(atol + rtol*max(fabs(y[idx]),
                                                     fabs(z[idx]))), 2);
    % endfor

        // Load and reduce along nupts (y direction), rows 1 to nrow
        for (int j = 1; j < nrow; j++)
        {
        % for k in range(ncola):
            idx = j*ldim + SOA_IX(i, ${k}, ${ncola});
            r = pow(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))), 2);
        % if norm == 'uniform':
            sdata${k}[tid] = max(sdata${k}[tid], r);
        % else:
            sdata${k}[tid] += r;
        % endif
        % endfor
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Unrolled reduction in local memory
    if (bid != gdim - 1)
    {
    % for n in pyfr.ilog2range(sharesz):
        if (tid < ${n})
        {
        % for k in range(ncola):
        % if norm == 'uniform':
            sdata${k}[tid] = max(sdata${k}[tid], sdata${k}[tid + ${n}]);
        % else:
            sdata${k}[tid] += sdata${k}[tid + ${n}];
        % endif
        % endfor
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    % endfor
    }
    // Last workgroup reduced with a variable sized loop
    else
    {
        for (int s = 1; s < lastblksize; s *= 2)
        {
            if (tid % (2*s) == 0 && tid + s < lastblksize)
            {
            % for k in range(ncola):
            % if norm == 'uniform':
                sdata${k}[tid] = max(sdata${k}[tid], sdata${k}[tid + s]);
            % else:
                sdata${k}[tid] += sdata${k}[tid + s];
            % endif
            % endfor
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
    }

    // Copy to global memory
    if (tid == 0)
    {
    % for k in range(ncola):
        err[${k}*gdim + bid] = sdata${k}[0];
    % endfor
    }
}
