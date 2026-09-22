<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    # One operand and one coefficient argument per register
    kargs = {'x0': f'inout fpdtype_t[{ncola}]'}
    kargs |= {f'x{i}': f'in fpdtype_t[{ncola}]' for i in range(1, nv)}
    kargs |= {f'a{i}': 'scalar fpdtype_t' for i in range(nv) if i != cidx}

    # Read the compensated operand's term and write that of the output, if any
    if cidx is not None:
        kargs[f'x{cidx}c'] = f'in fpdtype_t[{ncola}]'
    if wcomp:
        kargs['x0c'] = f'{'inout' if cidx == 0 else 'out'} fpdtype_t[{ncola}]'
%>

## Update each component from the operands weighted from index start
<%def name="stmts(start, op='=')">
% for k in range(ncola):
<%
    # Weight each operand other than the compensated one
    terms = []
    for l in range(start, nv):
        if l != cidx:
            isc = f'_in[{k}]*' if l in in_scale_idxs else ''
            terms.append(f'a{l}*{isc}x{l}[{k}]')

    # Sum the terms and apply any output scaling
    tsum = f'({' + '.join(terms) or '0'})'
    s = f'_out[{k}]*{tsum}' if out_scale else tsum

    # Accumulate the sum into x0, compensating if required
    if cidx is None:
        stmt = f'x0[{k}] {op} {s};'
    else:
        olo = f'x0c[{k}]' if wcomp else None
        stmt = pyfr.compadd(hi=f'x{cidx}[{k}]', lo=f'x{cidx}c[{k}]', inc=s,
                            ohi=f'x0[{k}]', olo=olo)
%>
    ${stmt}
% endfor
</%def>

<%pyfr:kernel name='axnpby' ndim='2' kargs='${kargs}'>
% if in_scale:
    const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
% endif
## Dispatch on the weight of x0 unless it is the compensated operand
% if cidx == 0:
    ${stmts(1)}
% else:
    if (a0 == 0.0)
    {
        ${stmts(1)}
    }
% if nv > 1 and cidx is None:
    else if (a0 == 1.0)
    {
        ${stmts(1, '+=')}
    }
% endif
    else
    {
        ${stmts(0)}
    }
% endif
</%pyfr:kernel>
