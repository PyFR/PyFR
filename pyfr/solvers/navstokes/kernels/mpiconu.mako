# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='mpiconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in mpi fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'>
    for (int i = 0; i < ${nvars}; i++)
    {
% if c['ldg-beta'] == -0.5:
        fpdtype_t con = ulin[i];
% elif c['ldg-beta'] == 0.5:
        fpdtype_t con = urin[i];
% else:
        fpdtype_t con = urin[i]*${0.5 + c['ldg-beta']}
                      + ulin[i]*${0.5 - c['ldg-beta']};
% endif

        ulout[i] = con;
    }
</%pyfr:kernel>

