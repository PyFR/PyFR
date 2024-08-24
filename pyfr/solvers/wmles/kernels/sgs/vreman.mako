<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if ndims == 2:
<%pyfr:macro name='eddy_viscosity' params='grad_uvw, delta, nu_sgs'>
    // Velocity derivatives
    fpdtype_t u_x = grad_uvw[0][0]; // g_xx
    fpdtype_t u_y = grad_uvw[1][0]; // g_yx
    fpdtype_t v_x = grad_uvw[0][1]; // g_xy
    fpdtype_t v_y = grad_uvw[1][1]; // g_yy

    // Compute beta_ij tensor components
    fpdtype_t beta_xx = u_x*u_x + u_y*u_y;
    fpdtype_t beta_yy = v_x*v_x + v_y*v_y;
    fpdtype_t beta_xy = u_x*u_y + v_x*v_y;

    // Compute B_mag
    fpdtype_t B_mag = beta_xx*beta_yy - beta_xy*beta_xy;

    // compute g_mag = g_ij*g_ij
    fpdtype_t g_mag = u_x*u_x + u_y*u_y 
                    + v_x*v_x + v_y*v_y;

    // Compute eddy viscosity
    fpdtype_t Csgs = sqrt(2.5)*0.16;
    fpdtype_t nu_sgs = pow(Csgs*delta, 2)*sqrt(B_mag/(g_mag + 1e-16));
</%pyfr:macro>

% elif ndims == 3:
<%pyfr:macro name='eddy_viscosity' params='grad_uvw, delta, nu_sgs'>
    // Velocity derivatives
    fpdtype_t u_x = grad_uvw[0][0]; // g_xx
    fpdtype_t u_y = grad_uvw[1][0]; // g_yx
    fpdtype_t u_z = grad_uvw[2][0]; // g_zx
    fpdtype_t v_x = grad_uvw[0][1]; // g_xy
    fpdtype_t v_y = grad_uvw[1][1]; // g_yy
    fpdtype_t v_z = grad_uvw[2][1]; // g_zy
    fpdtype_t w_x = grad_uvw[0][2]; // g_xz
    fpdtype_t w_y = grad_uvw[1][2]; // g_yz
    fpdtype_t w_z = grad_uvw[2][2]; // g_zz

    // Compute beta_ij tensor components
    fpdtype_t beta_xx = u_x*u_x + u_y*u_y + u_z*u_z;
    fpdtype_t beta_yy = v_x*v_x + v_y*v_y + v_z*v_z;
    fpdtype_t beta_zz = w_x*w_x + w_y*w_y + w_z*w_z;
    fpdtype_t beta_xy = u_x*u_y + v_x*v_y + w_x*w_y;
    fpdtype_t beta_yz = u_y*u_z + v_y*v_z + w_y*w_z;
    fpdtype_t beta_zx = u_z*u_x + v_z*v_x + w_z*w_x;

    // Compute B_mag
    fpdtype_t B_mag = beta_xx*beta_yy - beta_xy*beta_xy
                    + beta_xx*beta_zz - beta_zx*beta_zx
                    + beta_yy*beta_zz - beta_yz*beta_yz;

    // compute g_mag = g_ij*g_ij
    fpdtype_t g_mag = u_x*u_x + u_y*u_y + u_z*u_z
                    + v_x*v_x + v_y*v_y + v_z*v_z
                    + w_x*w_x + w_y*w_y + w_z*w_z;

    // Compute eddy viscosity
    fpdtype_t Csgs = sqrt(2.5)*0.16;
    fpdtype_t nu_sgs = pow(Csgs*delta, 2)*sqrt(B_mag/(g_mag + 1e-16));
</%pyfr:macro>
% endif
