<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    # Scale the input by variable
    xsum = 'isc*x' if in_scale else 'x'

    # Scale the staged output by variable
    ysum = 'osc*ys' if out_scale else 'ys'
%>

<%pyfr:kernel name='precondapply' ndim='1' coop='fit'
              minv='in tiled ${pcdtype}[${str(n)}][${str(n)}]'
              x='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              isc='in broadcast fpdtype_t[${str(nvars)}]'
              osc='in broadcast fpdtype_t[${str(nvars)}]'
              y='out fpdtype_t[${str(nupts)}][${str(nvars)}]'>
    // Stage the input of the item and make room for its output
    shared xs = @load(${xsum});
    shared fpdtype_t ys[${nupts}][${nvars}];
    @sync();

    // Each lane forms the inner product of its rows with the input
    for (int r = @lidx; r < ${n}; r += @lsz)
    {
        fpdtype_t yr = @dot(minv[r], xs);
        ys[r / ${nvars}][r % ${nvars}] = yr;
    }

    // Write out the output of the items of the block together
    @sync();
    @store(y, ${ysum});
</%pyfr:kernel>
