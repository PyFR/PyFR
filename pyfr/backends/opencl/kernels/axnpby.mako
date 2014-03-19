# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__kernel void
axnpby(int n,
       ${', '.join('__global fpdtype_t* restrict x' + str(i) for i in range(nv))},
       ${', '.join('fpdtype_t alpha' + str(i) for i in range(nv))})
{
    int strt = get_global_id(0);
    int incr = get_global_size(0);

    for (int i = strt; i < n; i += incr)
    {
        fpdtype_t axn = ${pyfr.dot('alpha{j}', 'x{j}[i]', j=(1, nv))};

        if (alpha0 == 0.0)
            x0[i] = axn;
        else if (alpha0 == 1.0)
            x0[i] += axn;
        else
            x0[i] = alpha0*x0[i] + axn;
    }
}
