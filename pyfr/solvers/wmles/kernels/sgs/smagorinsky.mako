<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if ndims == 2:
<%pyfr:macro name='eddy_viscosity' params='grad_uvw, delta, nu_sgs'>
    // Velocity derivatives
    fpdtype_t u_x = grad_uvw[0][0];
    fpdtype_t u_y = grad_uvw[1][0];
    fpdtype_t v_x = grad_uvw[0][1];
    fpdtype_t v_y = grad_uvw[1][1];

    // Compute strain rate tensor components
    fpdtype_t S_xx = u_x;
    fpdtype_t S_yy = v_y;
    fpdtype_t S_xy = 0.5*(u_y + v_x);

    // Compute |S| = sqrt(2*S_ij*S_ij)
    fpdtype_t S_mag = sqrt(2.0*(S_xx*S_xx + S_yy*S_yy + 2.0*S_xy*S_xy));

    // Compute eddy viscosity
    fpdtype_t Csgs = 0.16;
    fpdtype_t nu_sgs = pow(Csgs*delta, 2)*S_mag;
</%pyfr:macro>

% elif ndims == 3:
<%pyfr:macro name='eddy_viscosity' params='grad_uvw, delta, nu_sgs'>
    // Velocity derivatives (rho*grad[u,v,w])
    fpdtype_t u_x = grad_uvw[0][0];
    fpdtype_t u_y = grad_uvw[1][0];
    fpdtype_t u_z = grad_uvw[2][0];
    fpdtype_t v_x = grad_uvw[0][1];
    fpdtype_t v_y = grad_uvw[1][1];
    fpdtype_t v_z = grad_uvw[2][1];
    fpdtype_t w_x = grad_uvw[0][2];
    fpdtype_t w_y = grad_uvw[1][2];
    fpdtype_t w_z = grad_uvw[2][2];

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
    fpdtype_t Csgs = 0.16;
    fpdtype_t nu_sgs = pow(Csgs*delta, 2)*S_mag;
</%pyfr:macro>
% endif
