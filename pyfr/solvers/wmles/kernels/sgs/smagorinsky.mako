<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if ndims == 2:
<%pyfr:macro name='sub_grid_scale' params='uin, grad_uin, nu_sgs'>
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

    // Compute strain rate tensor components
    fpdtype_t S_xx = u_x;
    fpdtype_t S_yy = v_y;
    fpdtype_t S_xy = 0.5*(u_y + v_x);

    // Compute |S| = sqrt(2*S_ij*S_ij)
    fpdtype_t S_mag = sqrt(2.0*(S_xx*S_xx + S_yy*S_yy + 2.0*S_xy*S_xy));

    // Compute eddy viscosity
    fpdtype_t nu_sgs = ${sgs['Csgs']*sgs['Csgs']*sgs['delta']*sgs['delta']}*S_mag;
</%pyfr:macro>

% elif ndims == 3:
<%pyfr:macro name='sub_grid_scale' params='uin, grad_uin, nu_sgs'>
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

    // Compute strain rate tensor components
    fpdtype_t S_xx = u_x;
    fpdtype_t S_yy = v_y;
    fpdtype_t S_zz = w_z;
    fpdtype_t S_xy = 0.5*(u_y + v_x);
    fpdtype_t S_xz = 0.5*(u_z + w_x);
    fpdtype_t S_yz = 0.5*(v_z + w_y);

    // Compute |S| = sqrt(2*S_ij*S_ij)
    fpdtype_t S_mag = sqrt(2.0*(S_xx*S_xx + S_yy*S_yy + S_zz*S_zz 
                    + 2.0*(S_xy*S_xy + S_xz*S_xz + S_yz*S_yz)));
    
    // Compute eddy viscosity
    fpdtype_t nu_sgs = ${sgs['Csgs']*sgs['Csgs']*sgs['delta']*sgs['delta']}*S_mag;
</%pyfr:macro>
% endif
