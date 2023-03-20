<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

struct kargs
{
    int nrow;
    fpdtype_t ${','.join(f'*x{i}' for i in range(nv))};
    fpdtype_t ${','.join(f'a{i}' for i in range(nv))};
};

void axnpby(int ib, const struct kargs *restrict args)
{
    int nrow = args->nrow;
% for i in range(nv):
    fpdtype_t *x${i} = args->x${i}, a${i} = args->a${i};
% endfor

% if sorted(subdims) == list(range(ncola)):
    #pragma omp simd
    for (int i = ib*nrow*BLK_SZ*${ncola}; i < (ib + 1)*nrow*BLK_SZ*${ncola}; i++)
        x0[i] = ${pyfr.dot('a{l}', 'x{l}[i]', l=nv)};
% else:
    #define X_IDX_AOSOA(v, nv) ((_xi/SOA_SZ*(nv) + (v))*SOA_SZ + _xj)
    for (int _y = 0; _y < nrow; _y++)
    {
        for (int _xi = 0; _xi < BLK_SZ; _xi += SOA_SZ)
        {
            #pragma omp simd
            for (int _xj = 0; _xj < SOA_SZ; _xj++)
            {
                int i;

            % for k in subdims:
                i = _y*BLK_SZ*${ncola} + ib*BLK_SZ*${ncola}*nrow + X_IDX_AOSOA(${k}, ${ncola});
                x0[i] = ${pyfr.dot('a{l}', 'x{l}[i]', l=nv)};
            % endfor
            }
        }
    }
    #undef X_IDX_AOSOA
% endif
}
