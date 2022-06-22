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
    }

    // Perform the reduction inside of our work group
% if norm == 'uniform':
    acc = work_group_reduce_max(acc);
% else:
    acc = work_group_reduce_add(acc);
% endif

    // Copy to global memory
    if (tid == 0)
        reduced[k*gdim + bid] = acc;
}
