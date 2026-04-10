<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

struct kargs
{
    ixdtype_t nrow;
    fpdtype_t ${','.join(f'*x{i}' for i in range(nv))};
    fpdtype_t ${','.join(f'a{i}' for i in range(nv))};
};

void axnpby(int ib, const struct kargs *restrict args, int _disp_mask)
{
    ixdtype_t nrow = args->nrow;
% for i in range(nv):
    fpdtype_t *x${i} = args->x${i}, a${i} = args->a${i};
% endfor
% if in_scale or out_scale:
  % if in_scale:
    static const fpdtype_t _in[] = ${pyfr.carray(in_scale)};
  % endif
  % if out_scale:
    static const fpdtype_t _out[] = ${pyfr.carray(out_scale)};
  % endif

    #define X_IDX(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
    for (ixdtype_t _y = 0; _y < nrow; _y++)
    {
        for (ixdtype_t _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
        {
            #pragma omp simd
            for (ixdtype_t _xj = 0; _xj < SOA_SZ; _xj++)
            {
                ixdtype_t base = _y*BLK_SZ*${ncola} + ib*BLK_SZ*${ncola}*nrow;
            % for k in range(ncola):
                x0[base + X_IDX(${k}, ${ncola})] = ${pyfr.axnpby_expr(k, f'base + X_IDX({k}, {ncola})', 0, nv=nv, in_scale_idxs=in_scale_idxs, out_scale=out_scale)};
            % endfor
            }
        }
    }
    #undef X_IDX
% else:
    #pragma omp simd
    for (ixdtype_t i = ib*nrow*BLK_SZ*${ncola}; i < (ib + 1)*nrow*BLK_SZ*${ncola}; i++)
        x0[i] = ${pyfr.dot('a{l}', 'x{l}[i]', l=nv)};
% endif
}
