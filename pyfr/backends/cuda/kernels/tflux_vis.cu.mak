# -*- coding: utf-8 -*-

<%include file='idx_of.cu.mak' />
<%include file='flux_vis.cu.mak' />

/**
 * Computes the transformed viscous flux.
 */
__global__ void
tdisf_vis(int nupts, int neles,
          const ${dtype}* __restrict__ uin,
          const ${dtype}* __restrict__ smats,
          const ${dtype}* __restrict__ rcpdjacs,
          ${dtype}* __restrict__ tgrad_u,
          ${dtype} gamma, ${dtype} mu, ${dtype} rcppr,
          int ldu, int lds, int ldr, int ldg)
{
    int eidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (eidx < neles)
    {
        ${dtype} u[${nvars}], grad_u[${nvars}][${ndims}];
        ${dtype} s[${ndims}][${ndims}], f[${nvars}][${ndims}];

        for (int uidx = 0; uidx < nupts; ++uidx)
        {
            // Load in the solution
            for (int i = 0; i < ${nvars}; ++i)
                u[i] = uin[U_IDX_OF(uidx, eidx, i, neles, ldu)];

            // Load in the S-matrices
            for (int i = 0; i < ${ndims}; ++i)
                for (int j = 0; j < ${ndims}; ++j)
                    s[i][j] = smats[SMAT_IDX_OF(uidx, eidx, i, j, neles, lds)];

            // Get the reciprocal of the Jacobian
            ${dtype} rcpdjac = rcpdjacs[IDX_OF(uidx, eidx, ldr)];

            // Un-transform the solution gradient
            for (int j = 0; j < ${nvars}; ++j)
            {
            % for k in range(ndims):
                ${dtype} gu${k} = tgrad_u[F_IDX_OF(uidx, eidx, ${k}, j, nupts, neles, ldg)];
            % endfor

                for (int i = 0; i < ${ndims}; ++i)
                {
                    grad_u[j][i] = (${' + '.join('s[{0}][i]*gu{0}'.format(k)\
                                     for k in range(ndims))})
                                 * rcpdjac;
                }
            }

            // Compute the flux
            disf_vis(u, grad_u, f, NULL, NULL, gamma, mu, rcppr);

            // Transform and store
            for (int i = 0; i < ${ndims}; ++i)
                for (int j = 0; j < ${nvars}; ++j)
                {
                    int fidx = F_IDX_OF(uidx, eidx, i, j, nupts, neles, ldg);
                    tgrad_u[fidx] = ${' + '.join('s[i][{0}]*f[j][{0}]'.format(k)\
                                      for k in range(ndims))};
                }
        }
    }
}
