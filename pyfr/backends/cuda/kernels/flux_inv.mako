# -*- coding: utf-8 -*-

<%namespace name='util' module='pyfr.backends.cuda.makoutil' />
<%include file='idx_of' />
<%include file='flux_inv_impl' />

/**
 * Computes the transformed inviscid flux.
 */
__global__ void
tdisf_inv(int nupts, int neles,
          const ${dtype}* __restrict__ u,
          const ${dtype}* __restrict__ smats,
          ${dtype}* __restrict__ f,
          int ldu, int lds, int ldf)
{
    int eidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (eidx < neles)
    {
        ${dtype} uin[${nvars}], ftmp[${ndims}][${nvars}];

        for (int uidx = 0; uidx < nupts; ++uidx)
        {
            // Load in the soln
            for (int i = 0; i < ${nvars}; ++i)
                uin[i] = u[U_IDX_OF(uidx, eidx, i, neles, ldu)];

            // Compute the flux
            disf_inv_impl(uin, ftmp, NULL, NULL);

            // Transform and store
            for (int i = 0; i < ${ndims}; ++i)
            {
            % for k in range(ndims):
                ${dtype} s${k} = smats[SMAT_IDX_OF(uidx, eidx, i, ${k}, neles, lds)];
            % endfor

                for (int j = 0; j < ${nvars}; ++j)
                {
                    int fidx = F_IDX_OF(uidx, eidx, i, j, nupts, neles, ldf);
                    f[fidx] = ${util.dot('s{0}', 'ftmp[{0}][j]')};
                }
            }
        }
    }
}
