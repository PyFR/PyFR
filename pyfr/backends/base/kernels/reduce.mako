<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    # Declare an operand per matrix, a scalar per name and the combined result
    rows = f'[{nrow}]' if ndim == 1 else ''
    cols = f'[{ncola}]' if ncdim else ''
    kargs = {v: f'in fpdtype_t{rows}{cols}' for v in vvars}
    kargs |= {s: 'scalar fpdtype_t' for s in svars}
    kargs['_r'] = f'inout broadcast reduce({rop}) fpdtype_t[{nexprs}]'
%>

<%pyfr:kernel name='reduce' ndim='${ndim}' kargs='${kargs}'>
% for j in range(nexprs):
    _r[${j}] = ${ident};
% endfor

% if ndim == 1:
    // Combine every row of the column held by this lane
    for (int _i = 0; _i < ${nrow}; _i++)
    {
% endif
% for j, e in enumerate(exprs):
% for ek in e:
% if rop == 'sum':
        _r[${j}] += ${ek};
% else:
        _r[${j}] = f${rop}(_r[${j}], ${ek});
% endif
% endfor
% endfor
% if ndim == 1:
    }
% endif
</%pyfr:kernel>
