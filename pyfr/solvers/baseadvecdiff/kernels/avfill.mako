<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='avfill' ndim='1'
              vtx='in view(${str(nverts)}) fpdtype_t'
              artvisc_fpts='out fpdtype_t[${str(nfpts)}]'>
% for j, wts in enumerate(av_op):
    artvisc_fpts[${j}] = ${' + '.join(f'{w}*vtx[{i}]' for i, w in enumerate(wts) if w)};
% endfor
</%pyfr:kernel>
