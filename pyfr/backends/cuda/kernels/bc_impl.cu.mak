<%namespace name='util' module='pyfr.backends.cuda.makoutil' />

inline __device__ void
_bc_grad_u_zero(const ${dtype} ul[${ndims}],
                const ${dtype} grad_ul[${ndims}][${nvars}],
                ${dtype} grad_ur[${ndims}][${nvars}])
{
    for (int i = 0; i < ${ndims}; ++i)
        for (int j = 0; j < ${nvars}; ++j)
            grad_ur[i][j] = 0;
}

inline __device__ void
_bc_grad_u_copy(const ${dtype} ul[${ndims}],
                const ${dtype} grad_ul[${ndims}][${nvars}],
                ${dtype} grad_ur[${ndims}][${nvars}])
{
    for (int i = 0; i < ${ndims}; ++i)
        for (int j = 0; j < ${nvars}; ++j)
            grad_ur[i][j] = grad_ul[i][j];
}

% if bctype == 'isotherm-noslip':
inline __device__ void
bc_u_impl(const ${dtype} ul[${nvars}], ${dtype} ur[${nvars}])
{
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = -ul[${i + 1}];
% endfor
    ur[${nvars - 1}] = ${c['cpTw']|f}*ul[0]
                     + ${0.5|f}/ul[0]*${util.vlen('ul[{0}+1]')};
}

#define bc_grad_u_impl _bc_grad_u_copy
% elif bctype == 'sup-inflow':
<%
  rho, p = c['fs-rho'], c['fs-p']
  vv = [c['fs-%c' % v] for v in 'uvw'[:ndims]]
%>

inline __device__ void
bc_u_impl(const ${dtype} ul[${nvars}], ${dtype} ur[${nvars}])
{
    ur[0] = ${rho|f};
% for i, v in enumerate(vv):
    ur[${i + 1}] = ${rho*v|f};
% endfor
    ur[${nvars - 1}] = ${p/(c['gamma'] - 1) + 0.5*rho*sum(v**2 for v in vv)|f};
}

#define bc_grad_u_impl _bc_grad_u_zero
% elif bctype == 'sup-outflow':
inline __device__ void
bc_u_impl(const ${dtype} ul[${nvars}], ${dtype} ur[${nvars}])
{
% for i in range(nvars):
    ur[${i}] = ul[${i}];
% endfor
}

#define bc_grad_u_impl _bc_grad_u_copy
% elif bctype == 'sub-inflow':
<%
  rho, p = c['fs-rho'], c['fs-p']
  vv = [c['fs-%c' % v] for v in 'uvw'[:ndims]]
%>

inline __device__ void
bc_u_impl(const ${dtype} ul[${nvars}], ${dtype} ur[${nvars}])
{
    ur[0] = ${rho|f};
% for i, v in enumerate(vv):
    ur[${i + 1}] = ${rho*v|f};
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - ${0.5|f}/ul[0]*${util.vlen('ul[{0}+1]')}
                     + ${0.5*rho*sum(v**2 for v in vv)|f};
}

#define bc_grad_u_impl _bc_grad_u_zero
% elif bctype == 'sub-outflow':
inline __device__ void
bc_u_impl(const ${dtype} ul[${nvars}], ${dtype} ur[${nvars}])
{
% for i in range(ndims + 1):
    ur[${i}] = ul[${i}];
% endfor
    ur[${nvars - 1}] = ${c['fs-p']/(c['gamma'] - 1)|f}
                     + ${0.5|f}/ul[0]*${util.vlen('ul[{0}+1]')};
}

#define bc_grad_u_impl _bc_grad_u_copy
% endif
