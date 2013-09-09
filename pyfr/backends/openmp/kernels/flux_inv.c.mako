# -*- coding: utf-8 -*-

<%namespace name='util' module='pyfr.backends.openmp.makoutil' />
<%include file='common.h.mako' />
<%include file='flux_inv_impl.h.mako' />

static NOINLINE void
tdisf_inv_aux(int neles,
              ${util.arr_args('u', [nvars], const=True)},
              ${util.arr_args('smats', [ndims, ndims], const=True)},
              ${util.arr_args('f', [ndims, nvars])})
{
    ${util.arr_align('u', [nvars])};
    ${util.arr_align('smats', [ndims, ndims])};
    ${util.arr_align('f', [ndims, nvars])};

    for (int eidx = 0; eidx < neles; eidx++)
    {
        // Load in the components of the soln
        ${dtype} uin[${nvars}] = { ${', '.join('u{}[eidx]'.format(i)
                                               for i in range(nvars))} };
        ${dtype} ftmp[${ndims}][${nvars}];

        // Compute the flux
        disf_inv_impl(uin, ftmp, NULL, NULL);

        // Transform and store
    % for i, j in util.ndrange(ndims, nvars):
        f${i}${j}[eidx] = ${util.dot('smats{}{{0}}[eidx]'.format(i),
                                     'ftmp[{{0}}][{}]'.format(j))};
    % endfor
    }
}

void
tdisf_inv(int nupts, int neles,
          const ${dtype} *u, const ${dtype} *smats, ${dtype} *f,
          int lsdu, int lsds, int lsdf)
{
    #pragma omp parallel for
    for (int uidx = 0; uidx < nupts; uidx++)
    {
        tdisf_inv_aux(neles,
                      ${', '.join('u + (uidx*{} + {})*lsdu'.format(nvars, i)
                                  for i in range(nvars))},
                      ${', '.join('smats + (uidx*{} + {})*lsds'
                                  .format(ndims**2, i)
                                  for i in range(ndims**2))},
                      ${', '.join('f + (({}*nupts + uidx)*{} + {})*lsdf'
                                  .format(i, nvars, j)
                                  for i, j in util.ndrange(ndims, nvars))});
    }
}
