# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

void
axnpby(long *nrowp, long *ncolbp, long *ldimp, long *lsdimp,
       ${', '.join('fpdtype_t **xp' + str(i) for i in range(nv))},
       ${', '.join('double *ap' + str(i) for i in range(nv))})
{
    int ldim = *ldimp;
    int lsdim = *lsdimp;

% for i in range(nv):
    fpdtype_t *x${i} = *xp${i};
    fpdtype_t  a${i} = *ap${i};
% endfor

    #pragma omp parallel
    {
        int align = PYFR_ALIGN_BYTES / sizeof(fpdtype_t);
        int rb, re, cb, ce;
        loop_sched_2d(*nrowp, *ncolbp, align, &rb, &re, &cb, &ce);

        for (int r = rb; r < re; r++)
        {
        % for k in subdims:
            for (int i = cb; i < ce; i++)
            {
                int idx = i + ldim*r + ${k}*lsdim;
                fpdtype_t axn = ${pyfr.dot('a{l}', 'x{l}[idx]', l=(1, nv))};

                if (a0 == 0.0)
                    x0[idx] = axn;
                else if (a0 == 1.0)
                    x0[idx] += axn;
                else
                    x0[idx] = a0*x0[idx] + axn;
            }
        % endfor
        }
    }
}
