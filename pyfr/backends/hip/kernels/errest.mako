# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#define SQ(x) (x)*(x)

__global__ void
errest(int nrow, int ncolb, int ldim, fpdtype_t *__restrict__ err,
       fpdtype_t *__restrict__ x, fpdtype_t *__restrict__ y,
       fpdtype_t *__restrict__ z, fpdtype_t atol, fpdtype_t rtol)

{
    int tid = hipThreadIdx_x;
    int i = hipBlockIdx_x*hipBlockDim_x + tid;
    int lastblksize = ncolb % ${sharesz};

    __shared__ fpdtype_t sdata[${sharesz}];
    fpdtype_t r, acc = 0;

    if (i < ncolb)
    {
        for (int j = 0; j < nrow; j++)
        {
            int idx = j*ldim + SOA_IX(i, hipBlockIdx_y, hipGridDim_y);
            r = SQ(x[idx]/(atol + rtol*max(fabs(y[idx]), fabs(z[idx]))));
        % if norm == 'uniform':
            acc = max(r, acc);
        % else:
            acc += r;
        % endif
        }

        sdata[tid] = acc;
    }

    __syncthreads();

    // Unrolled reduction within full blocks
    if (hipBlockIdx_x != hipGridDim_x - 1)
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
        err[hipBlockIdx_y*hipGridDim_x + hipBlockIdx_x] = sdata[0];
}
