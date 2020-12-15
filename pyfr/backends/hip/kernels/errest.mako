# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
errest(int nrow, int ncolb, int ldim, fpdtype_t *__restrict__ err,
       fpdtype_t *__restrict__ x, fpdtype_t *__restrict__ y,
       fpdtype_t *__restrict__ z, fpdtype_t atol, fpdtype_t rtol)

{
    __shared__ fpdtype_t ${', '.join(f'sdata{i}[{sharesz}]'
                           for i in range(ncola))};
    fpdtype_t r;
    int idx;
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x*hipBlockDim_x + tid;
    int lastblksize = ncolb % hipBlockDim_x;

    if (i < ncolb)
    {
        // All threads load the values of the first row to shared memory
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

    __syncthreads();

    // Unrolled reduction within blocks
    if (hipBlockIdx_x != hipGridDim_x - 1)
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
            __syncthreads();
        }
    % endfor
    }
    // Last block reduced with a variable sized loop
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
                __syncthreads();
            }
        }
    }

    // Copy to global memory
    if (tid == 0)
    {
    % for k in range(ncola):
        err[${k}*hipGridDim_x + hipBlockIdx_x] = sdata${k}[0];
    % endfor
    }
}
