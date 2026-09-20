<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%!
def axnpby_expr(k, start, nv, in_scale_idxs, out_scale):
    terms = []
    for l in range(start, nv):
        coef, val = f'a{l}', f'x{l}[{k}]'
        if l in in_scale_idxs:
            terms.append(f'({coef})*_in[{k}]*({val})')
        else:
            terms.append(f'({coef})*({val})')

    if terms:
        expr = '(' + ' + '.join(terms) + ')'
        return f'_out[{k}]*{expr}' if out_scale else expr
    else:
        return '0'
%>

<%
    # One operand and one coefficient argument per register
    kargs = {'x0': f'inout fpdtype_t[{ncola}]'}
    kargs |= {f'x{i}': f'in fpdtype_t[{ncola}]' for i in range(1, nv)}
    kargs |= {f'a{i}': 'scalar fpdtype_t' for i in range(nv)}
%>

<%pyfr:kernel name='axnpby' ndim='2' kargs='${kargs}'>
% if in_scale:
    const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
% endif
% if out_scale:
    const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
% endif
    if (a0 == 0.0)
    {
% for k in range(ncola):
        x0[${k}] = ${axnpby_expr(k, 1, nv, in_scale_idxs, out_scale)};
% endfor
    }
% if nv > 1:
    else if (a0 == 1.0)
    {
% for k in range(ncola):
        x0[${k}] += ${axnpby_expr(k, 1, nv, in_scale_idxs, out_scale)};
% endfor
    }
% endif
    else
    {
% for k in range(ncola):
        x0[${k}] = ${axnpby_expr(k, 0, nv, in_scale_idxs, out_scale)};
% endfor
    }
</%pyfr:kernel>
