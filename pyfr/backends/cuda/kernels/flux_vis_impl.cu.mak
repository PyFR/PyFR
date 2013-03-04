# -*- coding: utf-8 -*-

% if ndims == 3:
/**
 * Computes the viscous flux and adds it to fout.
 */
inline __device__ void
disf_vis_impl_add(const ${dtype} uin[5], const ${dtype} grad_uin[3][5],
                  ${dtype} fout[3][5])
{
    ${dtype} rho  = uin[0];
    ${dtype} rhou = uin[1], rhov = uin[2], rhow = uin[3];
    ${dtype} E    = uin[4];

    ${dtype} rcprho = ${1.0|f}/rho;
    ${dtype} u = rcprho*rhou, v = rcprho*rhov, w = rcprho*rhow;

    ${dtype} rho_x = grad_uin[0][0];
    ${dtype} rho_y = grad_uin[1][0];
    ${dtype} rho_z = grad_uin[2][0];

    // Velocity derivatives (rho*grad[u,v,w])
    ${dtype} u_x = grad_uin[0][1] - u*rho_x;
    ${dtype} u_y = grad_uin[1][1] - u*rho_y;
    ${dtype} u_z = grad_uin[2][1] - u*rho_z;
    ${dtype} v_x = grad_uin[0][2] - v*rho_x;
    ${dtype} v_y = grad_uin[1][2] - v*rho_y;
    ${dtype} v_z = grad_uin[2][2] - v*rho_z;
    ${dtype} w_x = grad_uin[0][3] - w*rho_x;
    ${dtype} w_y = grad_uin[1][3] - w*rho_y;
    ${dtype} w_z = grad_uin[2][3] - w*rho_z;

    ${dtype} E_x = grad_uin[0][4];
    ${dtype} E_y = grad_uin[1][4];
    ${dtype} E_z = grad_uin[2][4];

    // Compute temperature derivatives (c_p*dT/d[x,y,z])
    ${dtype} T_x = rcprho*(E_x - (rcprho*rho_x*E + u*u_x + v*v_x + w*w_x));
    ${dtype} T_y = rcprho*(E_y - (rcprho*rho_y*E + u*u_y + v*v_y + w*w_y));
    ${dtype} T_z = rcprho*(E_z - (rcprho*rho_z*E + u*u_z + v*v_z + w*w_z));

    // Negated stress tensor elements
    ${dtype} t_xx = ${-2*mu|f}*rcprho*(u_x - ${1.0/3.0|f}*(u_x + v_y + w_z));
    ${dtype} t_yy = ${-2*mu|f}*rcprho*(v_y - ${1.0/3.0|f}*(u_x + v_y + w_z));
    ${dtype} t_zz = ${-2*mu|f}*rcprho*(w_z - ${1.0/3.0|f}*(u_x + v_y + w_z));
    ${dtype} t_xy = ${-mu|f}*rcprho*(v_x + u_y);
    ${dtype} t_xz = ${-mu|f}*rcprho*(u_z + w_x);
    ${dtype} t_yz = ${-mu|f}*rcprho*(w_y + v_z);

    fout[0][1] += t_xx;     fout[1][1] += t_xy;     fout[2][1] += t_xz;
    fout[0][2] += t_xy;     fout[1][2] += t_yy;     fout[2][2] += t_yz;
    fout[0][3] += t_xz;     fout[1][3] += t_yz;     fout[2][3] += t_zz;

    fout[0][4] += u*t_xx + v*t_xy + w*t_xz + ${-mu*gamma/pr|f}*T_x;
    fout[1][4] += u*t_xy + v*t_yy + w*t_yz + ${-mu*gamma/pr|f}*T_y;
    fout[2][4] += u*t_xz + v*t_yz + w*t_zz + ${-mu*gamma/pr|f}*T_z;
}
% endif
