# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ __launch_bounds__(${blocksz}) void
reduction(int nrow, int ncolb, int ldim, fpdtype_t *__restrict__ reduced,
          fpdtype_t *__restrict__ rcurr, fpdtype_t *__restrict__ rold,
% if method == 'errest':
          fpdtype_t *__restrict__ rerr, fpdtype_t atol, fpdtype_t rtol)
% elif method == 'resid' and dt_type == 'matrix':
          fpdtype_t *__restrict__ dt_mat, fpdtype_t dt_fac)
% elif method == 'resid':
          fpdtype_t dt_fac)
% endif
{
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + tid;
    int lastblksize = ncolb % ${blocksz};

    __shared__ fpdtype_t sdata[${blocksz}];
    fpdtype_t r, acc = 0;

    if (i < ncolb)
    {
        for (int j = 0; j < nrow; j++)
        {
            int idx = j*ldim + SOA_IX(i, blockIdx.y, gridDim.y);
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

    __syncthreads();

    // Unrolled reduction within full blocks
    if (blockIdx.x != gridDim.x - 1)
    {
    % for n in pyfr.ilog2range(blocksz):
        if (tid < ${n})
        {
        % if norm == 'uniform':
            sdata[tid] = max(sdata[tid], sdata[tid + ${n}]);
        % else:
            sdata[tid] += sdata[tid + ${n}];
        % endif
        }
        __syncthreads();
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
            __syncthreads();
        }
    }

    // Copy to global memory
    if (tid == 0)
        reduced[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0];
}
