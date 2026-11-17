<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% cyc = [(1, 2), (2, 0), (0, 1)][:ndims] %>

<%pyfr:macro name='nirf_source' params='t, u, ploc, src'
             externs='frame_omega_x, frame_omega_y, frame_omega_z,
                      frame_alpha_x, frame_alpha_y, frame_alpha_z,
                      frame_baccel_x, frame_baccel_y, frame_baccel_z'>
    fpdtype_t omega[3], alpha[3];
    fpdtype_t r[3] = {}, v[3] = {}, oxr[3] = {}, f[${ndims}];
    fpdtype_t invrho = 1.0/u[0];

    // Angular velocity and acceleration of the frame
% for i in range(3):
    omega[${i}] = ${frame_omega[i]};
    alpha[${i}] = ${frame_alpha[i]};
% endfor

    // Position relative to the centre of rotation and velocity
% for i in range(ndims):
    r[${i}] = ploc[${i}] - ${frame_origin[i]};
    v[${i}] = invrho*u[${i + 1}];
% endfor

    // Omega x r
% for i, (j, k) in enumerate(cyc):
    oxr[${i}] = omega[${j}]*r[${k}] - omega[${k}]*r[${j}];
% endfor

    // Centrifugal, Coriolis, Euler and linear acceleration forces
% for i, (j, k) in enumerate(cyc):
    f[${i}] = -u[0]*(omega[${j}]*oxr[${k}] - omega[${k}]*oxr[${j}]
                     + 2.0*(omega[${j}]*v[${k}] - omega[${k}]*v[${j}])
                     + alpha[${j}]*r[${k}] - alpha[${k}]*r[${j}]
                     + ${frame_baccel[i]});
    src[${i + 1}] += f[${i}];
% endfor

    // Work done by these forces
    src[${nvars - 1}] += ${pyfr.dot('f[{i}]', 'v[{i}]', i=ndims)};
</%pyfr:macro>
