<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% io, asn = ('inout', '+=') if add else ('out', '=') %>

<%pyfr:kernel name='physapply' ndim='1' coop='fit'
              op='in tiled fpdtype_t[${str(nrow)}][${str(ncol)}]'
              b='in fpdtype_t[${str(ncol)}][${str(nvars)}]'
              out='${io} fpdtype_t[${str(nrow)}][${str(nvars)}]'>
    // Stage the input and the output
    shared bs = @load(b);
% if add:
    shared fpdtype_t ys = @load(out);
% else:
    shared fpdtype_t ys[${nrow}][${nvars}];
% endif
    @sync();

    // Apply the rows of the operator this lane holds to the input
    for (int r = @lidx; r < ${nrow}; r += @lsz)
    {
        fpdtype_t t[${nvars}] = @dot(op[r], bs);
% for v in range(nvars):
        ys[r][${v}] ${asn} t[${v}];
% endfor
    }

    // Write the output
    @sync();
    @store(out, ys);
</%pyfr:kernel>
