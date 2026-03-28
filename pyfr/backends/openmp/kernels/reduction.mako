<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

struct kargs
{
    ixdtype_t nrow, nblocks;
    fpdtype_t *reduced;
    fpdtype_t ${', '.join(f'*{v}' for v in vvars)};
% if pvars:
    fpdtype_t (*_pv)[${ncola}];
% endif
% if svars:
    fpdtype_t ${', '.join(svars)};
% endif
};

void reduction(const struct kargs *restrict args)
{
    ixdtype_t nrow = args->nrow, nblocks = args->nblocks;
    fpdtype_t *reduced = args->reduced;
    fpdtype_t ${', '.join(f'*{v} = args->{v}' for v in vvars)};
% if pvars:
    const fpdtype_t (*_pv)[${ncola}] = args->_pv;
% endif
% for i, name in enumerate(pvars):
#define _pv_${name} _pv[${i}]
% endfor
% if svars:
    fpdtype_t ${', '.join(f'{s} = args->{s}' for s in svars)};
% endif

    // Initalise the reduction array
    fpdtype_t acc[${nexprs}] = ${pyfr.array(str(init_val), i=nexprs)};

    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)

% if rop == 'max':
    #pragma omp parallel for ${schedule} reduction(max : acc[:${nexprs}])
% else:
    #pragma omp parallel for ${schedule} reduction(+ : acc[:${nexprs}])
% endif
    for (ixdtype_t ib = 0; ib < nblocks; ib++)
    {
        for (ixdtype_t _y = 0; _y < nrow; _y++)
        {
            for (ixdtype_t _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
            {
                #pragma omp simd
                for (ixdtype_t _xj = 0; _xj < SOA_SZ; _xj++)
                {
                    for (ixdtype_t _k = 0; _k < ${ncola}; _k++)
                    {
                        ixdtype_t idx = (_y + ib*nrow)*BLK_SZ*${ncola} + X_IDX_AOSOA(_k, ${ncola});
                    % for j, e in enumerate(exprs):
                        % if rop == 'max':
                        acc[${j}] = max(acc[${j}], ${e});
                        % else:
                        acc[${j}] += ${e};
                        % endif
                    % endfor
                    }
                }
            }
        }
    }
    #undef X_IDX_AOSOA
% for name in pvars:
#undef _pv_${name}
% endfor

    // Copy
% for i in range(nexprs):
    reduced[${i}] = acc[${i}];
% endfor
}
