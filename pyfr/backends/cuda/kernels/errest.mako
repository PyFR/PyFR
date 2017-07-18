# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__global__ void
errest(int nrow, int ncolb, int ldim, fpdtype_t *__restrict__ err,
       fpdtype_t *__restrict__ x, fpdtype_t *__restrict__ y,
       fpdtype_t *__restrict__ z, fpdtype_t atol, fpdtype_t rtol)

{
    __shared__ fpdtype_t ${','.join('sdata{0}[{1}]'.format(i, sharesz) 
                                    for i in range(ncola))};
    fpdtype_t r;
    int idx;
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + tid;
    int lastblksize = ncolb % blockDim.x;

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
    if (blockIdx.x != gridDim.x - 1)
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
        err[${k}*gridDim.x + blockIdx.x] = sdata${k}[0];
    % endfor
    }
}
