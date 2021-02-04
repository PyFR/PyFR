# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#define SQ(x) (x)*(x)

__kernel void
errest(int nrow, int ncolb, int ldim, __global fpdtype_t* restrict err,
       ${', '.join(f'__global const fpdtype_t* restrict {i}' for i in 'xyz')},
       fpdtype_t atol, fpdtype_t rtol)

{
    int i = get_global_id(0), tid = get_local_id(0);
    int gdim = get_num_groups(0), bid = get_group_id(0);
    int ncola = get_num_groups(1), k = get_group_id(1);
    int lastblksize = ncolb % ${sharesz};

    __local fpdtype_t sdata[${sharesz}];
    fpdtype_t r, acc = 0;

    if (i < ncolb)
    {
        for (int j = 0; j < nrow; j++)
        {
            int idx = j*ldim + SOA_IX(i, k, ncola);
            r = SQ(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))));
        % if norm == 'uniform':
            acc = max(r, acc);
        % else:
            acc += r;
        % endif
        }

        sdata[tid] = acc;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Unrolled reduction within full blocks
    if (bid != gdim - 1)
    {
    % for n in pyfr.ilog2range(sharesz):
        if (tid < ${n})
        {
        % if norm == 'uniform':
            sdata[tid] = max(sdata[tid], sdata[tid + ${n}]);
        % else:
            sdata[tid] += sdata[tid + ${n}];
        % endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    % endfor
    }
    // Last block reduced with a variable sized loop
    else
    {
        for (int s = 1; s < lastblksize; s *= 2)
        {
            if (tid % (2*s) == 0 && tid + s < lastblksize)
            {
            % if norm == 'uniform':
                sdata[tid] = max(sdata[tid], sdata[tid + s]);
            % else:
                sdata[tid] += sdata[tid + s];
            % endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Copy to global memory
    if (tid == 0)
        err[k*gdim + bid] = sdata[0];
}
