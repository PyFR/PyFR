<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if ndims == 2:
<%pyfr:macro name='eddy_viscous_flux_add' params='uin, grad_uin, rcpdjac, fout'>
    fpdtype_t rho = uin[0], rhou = uin[1], rhov = uin[2], E = uin[3];

    fpdtype_t rcprho = 1.0/rho;
    fpdtype_t u = rcprho*rhou, v = rcprho*rhov;

    fpdtype_t rho_x = grad_uin[0][0];
    fpdtype_t rho_y = grad_uin[1][0];

    // Velocity derivatives (rho*grad[u,v])
    fpdtype_t u_x = grad_uin[0][1] - u*rho_x;
    fpdtype_t u_y = grad_uin[1][1] - u*rho_y;
    fpdtype_t v_x = grad_uin[0][2] - v*rho_x;
    fpdtype_t v_y = grad_uin[1][2] - v*rho_y;
    fpdtype_t grad_uvw[${ndims}][${ndims}] = {
        {u_x, v_x},
        {u_y, v_y}
    };

    fpdtype_t E_x = grad_uin[0][3];
    fpdtype_t E_y = grad_uin[1][3];

    // Compute temperature derivatives (c_v*dT/d[x,y])
    fpdtype_t T_x = rcprho*(E_x - (rcprho*rho_x*E + u*u_x + v*v_x));
    fpdtype_t T_y = rcprho*(E_y - (rcprho*rho_y*E + u*u_y + v*v_y));

    // Compute eddy viscosity
    fpdtype_t nu_sgs = 0;
    fpdtype_t delta = sqrt(1 / rcpdjac) / ${order + 1};
    ${pyfr.expand('eddy_viscosity', 'grad_uvw', 'delta', 'nu_sgs')};

    // Negated stress tensor elements
    fpdtype_t t_xx = -2*nu_sgs*(u_x - ${1.0/3.0}*(u_x + v_y));
    fpdtype_t t_yy = -2*nu_sgs*(v_y - ${1.0/3.0}*(u_x + v_y));
    fpdtype_t t_xy = -nu_sgs*(v_x + u_y);

    fout[0][1] += t_xx;     fout[1][1] += t_xy;
    fout[0][2] += t_xy;     fout[1][2] += t_yy;

    fout[0][3] += -nu_sgs*rho*${c['gamma']/c['Prt']}*T_x;
    fout[1][3] += -nu_sgs*rho*${c['gamma']/c['Prt']}*T_y;
</%pyfr:macro>

% elif ndims == 3:
<%pyfr:macro name='eddy_viscous_flux_add' params='uin, grad_uin, rcpdjac, fout'>
    fpdtype_t rho  = uin[0];
    fpdtype_t rhou = uin[1], rhov = uin[2], rhow = uin[3];
    fpdtype_t E    = uin[4];

    fpdtype_t rcprho = 1.0/rho;
    fpdtype_t u = rcprho*rhou, v = rcprho*rhov, w = rcprho*rhow;

    fpdtype_t rho_x = grad_uin[0][0];
    fpdtype_t rho_y = grad_uin[1][0];
    fpdtype_t rho_z = grad_uin[2][0];

    // Velocity derivatives (rho*grad[u,v,w])
    fpdtype_t u_x = grad_uin[0][1] - u*rho_x;
    fpdtype_t u_y = grad_uin[1][1] - u*rho_y;
    fpdtype_t u_z = grad_uin[2][1] - u*rho_z;
    fpdtype_t v_x = grad_uin[0][2] - v*rho_x;
    fpdtype_t v_y = grad_uin[1][2] - v*rho_y;
    fpdtype_t v_z = grad_uin[2][2] - v*rho_z;
    fpdtype_t w_x = grad_uin[0][3] - w*rho_x;
    fpdtype_t w_y = grad_uin[1][3] - w*rho_y;
    fpdtype_t w_z = grad_uin[2][3] - w*rho_z;
    fpdtype_t grad_uvw[${ndims}][${ndims}] = {
        {u_x, v_x, w_x},
        {u_y, v_y, w_y},
        {u_z, v_z, w_z}
    };

    fpdtype_t E_x = grad_uin[0][4];
    fpdtype_t E_y = grad_uin[1][4];
    fpdtype_t E_z = grad_uin[2][4];

    // Compute temperature derivatives (c_v*dT/d[x,y,z])
    fpdtype_t T_x = rcprho*(E_x - (rcprho*rho_x*E + u*u_x + v*v_x + w*w_x));
    fpdtype_t T_y = rcprho*(E_y - (rcprho*rho_y*E + u*u_y + v*v_y + w*w_y));
    fpdtype_t T_z = rcprho*(E_z - (rcprho*rho_z*E + u*u_z + v*v_z + w*w_z));

    // Compute eddy viscosity
    fpdtype_t nu_sgs = 0;
    fpdtype_t delta = cbrt(1 / rcpdjac) / ${order + 1};
    ${pyfr.expand('eddy_viscosity', 'grad_uvw', 'delta', 'nu_sgs')};

    // Negated stress tensor elements
    fpdtype_t t_xx = -2*nu_sgs*(u_x - ${1.0/3.0}*(u_x + v_y + w_z));
    fpdtype_t t_yy = -2*nu_sgs*(v_y - ${1.0/3.0}*(u_x + v_y + w_z));
    fpdtype_t t_zz = -2*nu_sgs*(w_z - ${1.0/3.0}*(u_x + v_y + w_z));
    fpdtype_t t_xy = -nu_sgs*(v_x + u_y);
    fpdtype_t t_xz = -nu_sgs*(u_z + w_x);
    fpdtype_t t_yz = -nu_sgs*(w_y + v_z);

    fout[0][1] += t_xx;     fout[1][1] += t_xy;     fout[2][1] += t_xz;
    fout[0][2] += t_xy;     fout[1][2] += t_yy;     fout[2][2] += t_yz;
    fout[0][3] += t_xz;     fout[1][3] += t_yz;     fout[2][3] += t_zz;

    fout[0][4] += -nu_sgs*rho*${c['gamma']/c['Prt']}*T_x;
    fout[1][4] += -nu_sgs*rho*${c['gamma']/c['Prt']}*T_y;
    fout[2][4] += -nu_sgs*rho*${c['gamma']/c['Prt']}*T_z;
</%pyfr:macro>
% endif