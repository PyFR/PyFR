<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% io, asn = ('inout', '+=') if add else ('out', '=') %>

<%pyfr:kernel name='physgradapply' ndim='1' coop='fit'
              op='in tiled fpdtype_t[${str(nupts)}][${str(ncol)}]'
              pnorm='in fpdtype_t[${str(ndims)}][${str(ncol)}]'
              ujump='in fpdtype_t[${str(ncol)}][${str(nvars)}]'
              out='${io} fpdtype_t[${str(ndims*nupts)}][${str(nvars)}]'>
    // Stage the common solution jumps and the output
    shared us = @load(ujump);
% if add:
    shared fpdtype_t ys = @load(out);
% else:
    shared fpdtype_t ys[${ndims*nupts}][${nvars}];
% endif
    @sync();

    // Form the rows of the correction this lane holds along each normal
    for (int r = @lidx; r < ${nupts}; r += @lsz)
    {
        fpdtype_t t[${ndims}][${nvars}] = @dot(pnorm*op[r], us);
% for k, v in pyfr.ndrange(ndims, nvars):
        ys[${k*nupts} + r][${v}] ${asn} t[${k}][${v}];
% endfor
    }

    // Write the correction
    @sync();
    @store(out, ys);
</%pyfr:kernel>
