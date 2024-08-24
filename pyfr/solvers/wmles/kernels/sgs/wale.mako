<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

% if ndims == 2:
<%pyfr:macro name='eddy_viscosity' params='grad_uvw, delta, nu_sgs'>
    // Velocity derivatives
    fpdtype_t u_x = grad_uvw[0][0]; // g_xx
    fpdtype_t u_y = grad_uvw[1][0]; // g_yx
    fpdtype_t v_x = grad_uvw[0][1]; // g_xy
    fpdtype_t v_y = grad_uvw[1][1]; // g_yy

    // Compute square of velocity gradient tensor
    fpdtype_t g2_xx = u_x*u_x + v_x*u_y;
    fpdtype_t g2_yy = u_y*v_x + v_y*v_y;
    fpdtype_t g2_xy = u_x*v_x + v_x*v_y;
    fpdtype_t g2_yx = u_y*u_x + v_y*u_y;

    // Compute traceless symmetric part of the square of the velocity gradient tensor
    fpdtype_t Sd_xx = 0.5*(g2_xx + g2_xx) - ${1.0/3.0}*(g2_xx + g2_yy);
    fpdtype_t Sd_xy = 0.5*(g2_xy + g2_yx);
    fpdtype_t Sd_yx = Sd_xy;
    fpdtype_t Sd_yy = 0.5*(g2_yy + g2_yy) - ${1.0/3.0}*(g2_xx + g2_yy);

    // Compute Sd_ij:Sd_ij
    fpdtype_t Sd_ij_Sd_ij = Sd_xx*Sd_xx + 2*Sd_xy*Sd_xy + Sd_yy*Sd_yy;

    // Compute square of strain rate tensor
    fpdtype_t S_xx = 0.5*(u_x + u_x);
    fpdtype_t S_xy = 0.5*(u_y + v_x);
    fpdtype_t S_yx = S_xy;
    fpdtype_t S_yy = 0.5*(v_y + v_y);

    // Compute S_ij:S_ij (strain rate tensor)
    fpdtype_t S_ij_S_ij = S_xx*S_xx + 2*S_xy*S_xy + S_yy*S_yy;

    // Compute eddy viscosity
    fpdtype_t Csgs = 0.325;
    fpdtype_t num = pow(Sd_ij_Sd_ij, 1.5);
    fpdtype_t den = pow(S_ij_S_ij, 2.5) + pow(Sd_ij_Sd_ij, 1.25);
    fpdtype_t nu_sgs = pow(Csgs*delta, 2) * num / (den + 1e-16);
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

    // Compute square of velocity gradient tensor
    fpdtype_t g2_xx = u_x*u_x + v_x*u_y + w_x*u_z;
    fpdtype_t g2_xy = u_x*v_x + v_x*v_y + w_x*v_z;
    fpdtype_t g2_xz = u_x*w_x + v_x*w_y + w_x*w_z;
    fpdtype_t g2_yx = u_y*u_x + v_y*u_y + w_y*u_z;
    fpdtype_t g2_yy = u_y*v_x + v_y*v_y + w_y*v_z;
    fpdtype_t g2_yz = u_y*w_x + v_y*w_y + w_y*w_z;
    fpdtype_t g2_zx = u_z*u_x + v_z*u_y + w_z*u_z;
    fpdtype_t g2_zy = u_z*v_x + v_z*v_y + w_z*v_z;
    fpdtype_t g2_zz = u_z*w_x + v_z*w_y + w_z*w_z;

    // Compute traceless symmetric part of the square of the velocity gradient tensor
    fpdtype_t Sd_xx = 0.5*(g2_xx + g2_xx) - ${1.0/3.0}*(g2_xx + g2_yy + g2_zz);
    fpdtype_t Sd_xy = 0.5*(g2_xy + g2_yx);
    fpdtype_t Sd_xz = 0.5*(g2_xz + g2_zx);
    fpdtype_t Sd_yx = Sd_xy;
    fpdtype_t Sd_yy = 0.5*(g2_yy + g2_yy) - ${1.0/3.0}*(g2_xx + g2_yy + g2_zz);
    fpdtype_t Sd_yz = 0.5*(g2_yz + g2_zy);
    fpdtype_t Sd_zx = Sd_xz;
    fpdtype_t Sd_zy = Sd_yz;
    fpdtype_t Sd_zz = 0.5*(g2_zz + g2_zz) - ${1.0/3.0}*(g2_xx + g2_yy + g2_zz);

    // Compute Sd_ij:Sd_ij
    fpdtype_t Sd_ij_Sd_ij = Sd_xx*Sd_xx + Sd_yy*Sd_yy + Sd_zz*Sd_zz + 
                            2*(Sd_xy*Sd_xy + Sd_yz*Sd_yz + Sd_zx*Sd_zx);

    // Compute square of strain rate tensor
    fpdtype_t S_xx = 0.5*(u_x + u_x);
    fpdtype_t S_xy = 0.5*(u_y + v_x);
    fpdtype_t S_xz = 0.5*(u_z + w_x);
    fpdtype_t S_yx = S_xy;
    fpdtype_t S_yy = 0.5*(v_y + v_y);
    fpdtype_t S_yz = 0.5*(v_z + w_y);
    fpdtype_t S_zx = S_xz;
    fpdtype_t S_zy = S_yz;
    fpdtype_t S_zz = 0.5*(w_z + w_z);

    // Compute S_ij:S_ij (strain rate tensor)
    fpdtype_t S_ij_S_ij = S_xx*S_xx + S_yy*S_yy + S_zz*S_zz + 
                          2*(S_xy*S_xy + S_yz*S_yz + S_zx*S_zx);

    // Compute eddy viscosity
    fpdtype_t Csgs = 0.325;
    fpdtype_t num = pow(Sd_ij_Sd_ij, 1.5);
    fpdtype_t den = pow(S_ij_S_ij, 2.5) + pow(Sd_ij_Sd_ij, 1.25);
    fpdtype_t nu_sgs = pow(Csgs*delta, 2) * num / (den + 1e-16);
</%pyfr:macro>
% endif