# -*- coding: utf-8 -*-

% if ndims == 3:
/**
 * Computes the viscous flux.
 */
inline __device__ void
disf_vis(const ${dtype} uin[5], const ${dtype} grad_uin[5][3],
         ${dtype} fout[5][3], ${dtype} gamma, ${dtype} mu, ${dtype} pr)
{
    ${dtype} rho  = uin[0];
    ${dtype} rhou = uin[1], rhov = uin[2], rhow = uin[3];
    ${dtype} E    = uin[4];

    ${dtype} rcprho = 1.0/rho;
    ${dtype} u = rcprho*rhou, v = rcprho*rhov, w = rcprho*rhow;

    ${dtype} rho_x = grad_uin[0][0];
    ${dtype} rho_y = grad_uin[0][1];
    ${dtype} rho_z = grad_uin[0][2];

    // Velocity derivs (can be optimized!)
    ${dtype} u_x = rcprho*(grad_uin[1][0] - u*rho_x);
    ${dtype} u_y = rcprho*(grad_uin[1][1] - u*rho_y);
    ${dtype} u_z = rcprho*(grad_uin[1][2] - u*rho_z);
    ${dtype} v_x = rcprho*(grad_uin[2][0] - v*rho_x);
    ${dtype} v_y = rcprho*(grad_uin[2][1] - v*rho_y);
    ${dtype} v_z = rcprho*(grad_uin[2][2] - v*rho_z);
    ${dtype} w_x = rcprho*(grad_uin[3][0] - w*rho_x);
    ${dtype} w_y = rcprho*(grad_uin[3][1] - w*rho_y);
    ${dtype} w_z = rcprho*(grad_uin[3][2] - w*rho_z);

    ${dtype} E_x = grad_uin[4][0];
    ${dtype} E_y = grad_uin[4][1];
    ${dtype} E_z = grad_uin[4][2];

    // Stress tensor (can be optimized, mu = -mu!)
    ${dtype} t_xx = 2*mu*(u_x - (1.0/3.0)*(u_x + v_y + w_z));
    ${dtype} t_yy = 2*mu*(v_y - (1.0/3.0)*(u_x + v_y + w_z));
    ${dtype} t_zz = 2*mu*(w_z - (1.0/3.0)*(u_x + v_y + w_z));
    ${dtype} t_xy = mu*(v_x + u_y);
    ${dtype} t_xz = mu*(u_z + w_x);
    ${dtype} t_yz = mu*(w_y + v_z);

    // Compute the pressure
    ${dtype} p = (gamma - 1.0)*(E - 0.5*(rhou*u + rhov*v + rhow*w));

    // Compute temperature derivatives (c_p*dT/d[x,y,z])
    ${dtype} T_x = rcprho*(E_x - rcprho*rho_x*E) - (u*u_x + v*v_x + w*w_x);
    ${dtype} T_y = rcprho*(E_y - rcprho*rho_y*E) - (u*u_y + v*v_y + w*w_y);
    ${dtype} T_z = rcprho*(E_z - rcprho*rho_z*E) - (u*u_z + v*v_z + w*w_z);

    fout[0][0] = rhou;
    fout[0][1] = rhov;
    fout[0][2] = rhow;

    fout[1][0] = rhou*u + p - t_xx;
    fout[1][1] = rhov*u - t_xy;
    fout[1][2] = rhow*u - t_xz;

    fout[2][0] = rhou*v - t_xy;
    fout[2][1] = rhov*v + p - t_yy;
    fout[2][2] = rhow*v - t_yz;

    fout[3][0] = rhou*w - t_xz;
    fout[3][1] = rhov*w - t_yz;
    fout[3][2] = rhow*w + p - t_zz;

    fout[4][0] = (E + p)*u - (u*t_xx + v*t_xy + w*t_xz + mu*gamma/pr*T_x);
    fout[4][1] = (E + p)*v - (u*t_xy + v*t_yy + w*t_yz + mu*gamma/pr*T_y);
    fout[4][2] = (E + p)*w - (u*t_xz + v*t_yz + w*t_zz + mu*gamma/pr*T_z);
}
% endif
