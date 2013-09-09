# -*- coding: utf-8 -*-

<%namespace name='util' module='pyfr.backends.openmp.makoutil' />
<%include file='common.h.mako' />

static NOINLINE void
negdivconf_aux(int neles,
               ${util.arr_args('tdivtconf', [nvars])},
               const ${dtype} *restrict rcpdjac)
{
    ${util.arr_align('tdivtconf', [nvars])};
    ASSUME_ALIGNED(rcpdjac);

    for (int eidx = 0; eidx < neles; eidx++)
    {
    % for i in range(nvars):
        tdivtconf${i}[eidx] *= -rcpdjac[eidx];
    % endfor
    }
}

void
negdivconf(int nupts, int neles,
           ${dtype} *tdivtconf, const ${dtype} *rcpdjac,
           int ldr, int lsdt)
{
    #pragma omp parallel for
    for (int uidx = 0; uidx < nupts; uidx++)
    {
        negdivconf_aux(neles,
                       ${', '.join('tdivtconf + (uidx*{} + {})*lsdt'
                                   .format(nvars, i)
                                   for i in range(nvars))},
                       rcpdjac + uidx*ldr);
    }
}
