<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='source' params='t, u, ploc, src'>
% for i, ex in enumerate(src_exprs):
    src[${i}] += ${ex};
% endfor
</%pyfr:macro>
