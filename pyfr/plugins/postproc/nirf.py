from ast import literal_eval

import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin


def _quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


class NIRFPostProc(BasePostProcPlugin):
    name = 'nirf'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'
    transform = True
    source_prefix = 'soln'

    def _process(self, data):
        ndims = data.ndims

        sdata = data.soln.state.get('plugins/nirf')
        if sdata is None:
            raise RuntimeError('No NIRF state in solution file')

        fquat = sdata['quat']
        fomega = sdata['omega']
        fvelo = sdata['velo']
        floc = sdata['loc']

        fx0 = np.zeros(3)
        fx0[:ndims] = literal_eval(
            data.cfg.get('solver-plugin-nirf', 'center-of-rot')
        )

        R = _quat_to_rotmat(fquat)

        # r_body (before rotation); pad to 3D for omega x r
        r = np.zeros((3, *data.ploc.shape[1:]))
        r[:ndims] = data.ploc - fx0[:ndims, None, None]

        # Transform coordinates: x_lab = R*(x_body - x0) + x0 + loc
        ploc = data.ploc
        ploc -= fx0[:ndims, None, None]
        ploc[:] = np.tensordot(R[:ndims, :ndims], ploc, axes=1)
        ploc += fx0[:ndims, None, None]

        apply_trans = self.cfg.getbool(self.cfgsect,
                                       'apply-translation', False)
        if apply_trans:
            ploc += floc[:ndims, None, None]

        # Transform velocity: u_lab = R*(u_body + omega x r_body) + V_frame
        vs = data.pris[1:ndims + 1]
        u_body = np.zeros_like(r)
        u_body[:ndims] = vs

        oxr = np.cross(fomega, r, axisb=0, axisc=0)
        u_lab = np.tensordot(R, u_body + oxr, axes=1) + fvelo[:, None, None]

        for d, v in enumerate(vs):
            v[:] = u_lab[d]

        if not data.has_grads:
            return

        # Scalar gradients (rho, p) rotate as covectors: grad_lab = R*grad_body
        Rd = R[:ndims, :ndims]
        for vi in (0, ndims + 1):
            data.grad_pris[vi][:] = np.tensordot(Rd, data.grad_pris[vi],
                                                 axes=1)

        # Velocity gradient tensor: G_lab = R*(G_body + [Omega x])*R^T
        # Build padded (3, 3, npts, neles) so the omega correction works in 2D
        G = np.zeros((3, 3, *r.shape[1:]))
        for i in range(ndims):
            G[i, :ndims] = data.grad_pris[1 + i]

        wx, wy, wz = fomega
        G += np.array([[0, -wz,  wy],
                       [wz,  0, -wx],
                       [-wy, wx,  0]])[:, :, None, None]

        G_lab = np.einsum('ik,kl...,jl->ij...', R, G, R)

        for i in range(ndims):
            data.grad_pris[1 + i][:] = G_lab[i, :ndims]
