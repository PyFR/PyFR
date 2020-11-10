# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.backends.base.makocommon.transform'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    fpdtype_t utl[${nvars}], utr[${nvars}], ntf[${nvars}];

    utl[0] = ul[0]; utl[${nvars-1}] = ul[${nvars-1}];
    utr[0] = ur[0]; utr[${nvars-1}] = ur[${nvars-1}];

    ${pyfr.expand('transform_to','n', 'ul', 'utl','1')};
    ${pyfr.expand('transform_to','n', 'ur', 'utr','1')};

    ${pyfr.expand('rsolve_t1d','utl','utr','ntf')};

    nf[0] = ntf[0]; nf[${nvars-1}] = ntf[${nvars-1}];
    ${pyfr.expand('transform_from','n','ntf','nf','1')};

</%pyfr:macro>
