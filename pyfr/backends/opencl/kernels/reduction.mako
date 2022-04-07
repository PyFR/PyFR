# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__kernel void
reduction(int nrow, int ncolb, int ldim, __global fpdtype_t* restrict reduced,
          __global const fpdtype_t* restrict rcurr,
          __global const fpdtype_t* restrict rold,
% if method == 'errest':
          __global const fpdtype_t* restrict rerr, fpdtype_t atol, fpdtype_t rtol)
% elif method == 'resid' and dt_type == 'matrix':
          __global const fpdtype_t* restrict dt_mat, fpdtype_t dt_fac)
% elif method == 'resid':
          fpdtype_t dt_fac)
% endif
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

        sdata[tid] = acc;
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

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
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
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
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Copy to global memory
    if (tid == 0)
        reduced[k*gdim + bid] = sdata[0];
}
