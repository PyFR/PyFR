# -*- coding: utf-8 -*-

<%namespace name='util' module='pyfr.backends.c.makoutil' />
<%include file='common.h.mak' />
<%include file='flux_inv_impl.h.mak' />
<%include file='flux_vis_impl.h.mak' />

static NOINLINE void
tdisf_vis_aux(size_t neles,
              ${util.arr_args('u', [nvars], const=True)},
              ${util.arr_args('sm', [ndims, ndims], const=True)},
              const ${dtype} *restrict rcpdjac,
              ${util.arr_args('tgrad_u', [ndims, nvars])})
{
    ${util.arr_align('u', [nvars])};
    ${util.arr_align('sm', [ndims, ndims])};
    ASSUME_ALIGNED(rcpdjac);
    ${util.arr_align('tgrad_u', [ndims, nvars])};

    for (size_t eidx = 0; eidx < neles; eidx++)
    {
        ${dtype} u[${nvars}], grad_u[${ndims}][${nvars}], f[${ndims}][${nvars}];

        // Load in the soln
    % for i in range(nvars):
        u[${i}] = u${i}[eidx];
    % endfor

        // Load and un-transform the soln gradient
    % for i, j in util.ndrange(ndims, nvars):
        grad_u[${i}][${j}] = ${util.dot('sm{{0}}{}[eidx]'.format(i),
                                        'tgrad_u{{0}}{}[eidx]'.format(j))}
                           * rcpdjac[eidx];
    % endfor

        // Compute the flux (F = Fi + Fv)
        disf_inv_impl(u, f, NULL, NULL);
        disf_vis_impl_add(u, grad_u, f);

        // Transform and store
    % for i, j in util.ndrange(ndims, nvars):
        tgrad_u${i}${j}[eidx] = ${util.dot('sm{}{{0}}[eidx]'.format(i),
                                           'f[{{0}}][{}]'.format(j))};
    % endfor
    }
}

void
tdisf_vis(size_t nupts, size_t neles,
          const ${dtype} *u, const ${dtype} *smats,
          const ${dtype} *rcpdjac, ${dtype} *tgrad_u,
          size_t ldu, size_t lds, size_t ldr, size_t ldg,
          size_t lsdu, size_t lsds, size_t lsdg)
{
    #pragma omp parallel for
    for (size_t uidx = 0; uidx < nupts; uidx++)
    {
        tdisf_vis_aux(neles,
                      ${', '.join('u + uidx*ldu + {}*lsdu'.format(i)
                                  for i in range(nvars))},
                      ${', '.join('smats + uidx*lds + {}*lsds'.format(i)
                                  for i in range(ndims**2))},
                      rcpdjac + uidx*ldr,
                      ${', '.join('tgrad_u + ({}*nupts + uidx)*ldg + {}*lsdg'
                                  .format(i, j)
                                  for i, j in util.ndrange(ndims, nvars))});
    }
}
