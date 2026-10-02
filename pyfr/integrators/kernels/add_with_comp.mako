<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%
    # Output register and its compensation term, if any
    kargs = {'x0': f'inout fpdtype_t[{ncola}]'}
    if ycomp:
        kargs['x0c'] = f'inout fpdtype_t[{ncola}]'

    # Compensated register and its term unless the output is updated in-place
    if not inplace:
        kargs['u'] = kargs['uc'] = f'in fpdtype_t[{ncola}]'

    # One weight per register and one operand per register not aliasing x0
    xn = ['x0' if i in oidx else f'x{i}' for i in range(nv)]
    kargs |= {f'a{i}': 'scalar fpdtype_t' for i in range(1, nv)}
    kargs |= {x: f'in fpdtype_t[{ncola}]' for x in xn[1:] if x != 'x0'}

    # Take the compensated register from the output when updating in-place
    u, uc = ('x0', 'x0c') if inplace else ('u', 'uc')
%>

<%pyfr:kernel name='add_with_comp' ndim='2' kargs='${kargs}'>
% for k in range(ncola):
<%
    hi, lo = f'{u}[{k}]', f'{uc}[{k}]'
    inc = ' + '.join(f'a{i}*{xn[i]}[{k}]' for i in range(1, nv))
    olo = f'x0c[{k}]' if ycomp else None
%>
    ${pyfr.compadd(hi=hi, lo=lo, inc=inc, ohi=f'x0[{k}]', olo=olo)}
% endfor
</%pyfr:kernel>
