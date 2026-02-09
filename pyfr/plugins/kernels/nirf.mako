<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='nirf_source_2d' params='t, u, ploc, src'
             externs='frame_omega_z, frame_alpha_z, frame_accel_x, frame_accel_y'>
## Extract primitive variables
    fpdtype_t rho = u[0];
    fpdtype_t invrho = 1.0/rho;
    fpdtype_t vx = u[1]*invrho;
    fpdtype_t vy = u[2]*invrho;

## Position relative to frame origin
    fpdtype_t rx = ploc[0] - ${frame_origin_x};
    fpdtype_t ry = ploc[1] - ${frame_origin_y};

## Momentum source: centrifugal + coriolis + euler + linear (z-components only)
    fpdtype_t Sx = rho*(${frame_omega_z_expr}*${frame_omega_z_expr}*rx + 2.0*${frame_omega_z_expr}*vy + ${frame_alpha_z_expr}*ry) - rho*${frame_accel_x_expr};
    fpdtype_t Sy = rho*(${frame_omega_z_expr}*${frame_omega_z_expr}*ry - 2.0*${frame_omega_z_expr}*vx - ${frame_alpha_z_expr}*rx) - rho*${frame_accel_y_expr};

## Energy source: work done by fictitious forces
    src[1] += Sx;
    src[2] += Sy;
    src[3] += Sx*vx + Sy*vy;
</%pyfr:macro>

<%pyfr:macro name='nirf_source_3d' params='t, u, ploc, src'
             externs='frame_omega_x, frame_omega_y, frame_omega_z, frame_alpha_x, frame_alpha_y, frame_alpha_z,
                      frame_accel_x, frame_accel_y, frame_accel_z'>
## Extract primitive variables
    fpdtype_t rho = u[0];
    fpdtype_t invrho = 1.0/rho;
    fpdtype_t vx = u[1]*invrho;
    fpdtype_t vy = u[2]*invrho;
    fpdtype_t vz = u[3]*invrho;

## Position relative to frame origin
    fpdtype_t rx = ploc[0] - ${frame_origin_x};
    fpdtype_t ry = ploc[1] - ${frame_origin_y};
    fpdtype_t rz = ploc[2] - ${frame_origin_z};

## Omega x r
    fpdtype_t OxR_x = ${frame_omega_y_expr}*rz - ${frame_omega_z_expr}*ry;
    fpdtype_t OxR_y = ${frame_omega_z_expr}*rx - ${frame_omega_x_expr}*rz;
    fpdtype_t OxR_z = ${frame_omega_x_expr}*ry - ${frame_omega_y_expr}*rx;

## Centrifugal: Omega x (Omega x r)
    fpdtype_t cent_x = ${frame_omega_y_expr}*OxR_z - ${frame_omega_z_expr}*OxR_y;
    fpdtype_t cent_y = ${frame_omega_z_expr}*OxR_x - ${frame_omega_x_expr}*OxR_z;
    fpdtype_t cent_z = ${frame_omega_x_expr}*OxR_y - ${frame_omega_y_expr}*OxR_x;

## Coriolis: 2 * Omega x v
    fpdtype_t cori_x = 2.0*(${frame_omega_y_expr}*vz - ${frame_omega_z_expr}*vy);
    fpdtype_t cori_y = 2.0*(${frame_omega_z_expr}*vx - ${frame_omega_x_expr}*vz);
    fpdtype_t cori_z = 2.0*(${frame_omega_x_expr}*vy - ${frame_omega_y_expr}*vx);

## Euler: alpha x r
    fpdtype_t euler_x = ${frame_alpha_y_expr}*rz - ${frame_alpha_z_expr}*ry;
    fpdtype_t euler_y = ${frame_alpha_z_expr}*rx - ${frame_alpha_x_expr}*rz;
    fpdtype_t euler_z = ${frame_alpha_x_expr}*ry - ${frame_alpha_y_expr}*rx;

## Total momentum source
    fpdtype_t Sx = rho*(cent_x + cori_x + euler_x) - rho*${frame_accel_x_expr};
    fpdtype_t Sy = rho*(cent_y + cori_y + euler_y) - rho*${frame_accel_y_expr};
    fpdtype_t Sz = rho*(cent_z + cori_z + euler_z) - rho*${frame_accel_z_expr};

## Add to source array
    src[1] += Sx;
    src[2] += Sy;
    src[3] += Sz;
    src[4] += Sx*vx + Sy*vy + Sz*vz;
</%pyfr:macro>
