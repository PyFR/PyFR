# -*- coding: utf-8 -*-

<%include file='idx_of.cu.mako' />

__global__ void
gradcoru(int nfpts, int neles,
         const ${dtype}* __restrict__ jmats,
         ${dtype}* __restrict__ tgrad_uinout,
         int ldj, int ldg)
{
    int eidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (eidx < neles)
    {
        ${dtype} jm[${ndims}][${ndims}];

        for (int fidx = 0; fidx < nfpts; ++fidx)
        {
            // Load in the J-matrices
            for (int i = 0; i < ${ndims}; ++i)
                for (int j = 0; j < ${ndims}; ++j)
                    jm[i][j] = jmats[JMAT_IDX_OF(fidx, eidx, i, j, neles, ldj)];

            // Un-transform the solution gradient
            for (int j = 0; j < ${nvars}; ++j)
            {
                // Get the indices
            % for i in range(ndims):
                int gidx${i} = GRAD_U_IDX_OF(fidx, eidx, ${i}, j, nfpts, neles, ldg);
            % endfor

                // Load the solution gradients
            % for i in range(ndims):
                ${dtype} gu${i} = tgrad_uinout[gidx${i}];
            % endfor

                // Transform and store
            % for i in range(ndims):
                tgrad_uinout[gidx${i}] = ${' + '.join('jm[{0}][{1}]*gu{1}'.format(i, k)
                                           for k in range(ndims))};
            % endfor
            }
        }
    }
}
