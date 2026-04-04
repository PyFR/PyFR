<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

__kernel void
reduction(ixdtype_t nrow, ixdtype_t ncolb, ixdtype_t ldim,
          __global fpdtype_t* restrict reduced,
% for v in vvars:
          __global fpdtype_t* restrict ${v}${',' if not loop.last or svars else ')'}
% endfor
% for s in svars:
           fpdtype_t ${s}${')' if loop.last else ','}
% endfor
{
    ixdtype_t i = get_global_id(0), tid = get_local_id(0);
    ixdtype_t gdim = get_num_groups(0), bid = get_group_id(0);
    ixdtype_t ncola = get_num_groups(1), k = get_group_id(1);
% for i, name in enumerate(pvars):
    const __global fpdtype_t *_pv_${name} = _pv + ${i}*ncola;
% endfor

% for ei in range(nexprs):
    fpdtype_t acc_${ei} = ${init_val};
% endfor

    if (i < ncolb)
    {
        for (ixdtype_t j = 0; j < nrow; j++)
        {
            ixdtype_t idx = j*ldim + SOA_IX(i, k, ncola);
% for ei in range(nexprs):
  % if rop == 'max':
            acc_${ei} = max(acc_${ei}, ${exprs[ei]});
  % else:
            acc_${ei} += ${exprs[ei]};
  % endif
% endfor
        }
    }

<%
    rop_fn = 'max' if rop == 'max' else 'add'
%>
% for ei in range(nexprs):
    acc_${ei} = work_group_reduce_${rop_fn}(acc_${ei});
% endfor

    if (tid == 0)
    {
% for ei in range(nexprs):
        reduced[${ei}*ncola*gdim + k*gdim + bid] = acc_${ei};
% endfor
    }
}
