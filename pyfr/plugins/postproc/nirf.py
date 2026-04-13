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

    def process(self, data):
        ndims = data.ndims
        soln = data.soln

        sdata = soln.state.get('plugins/nirf')
        if sdata is None:
            return

        fquat = sdata['quat']
        fomega = sdata['omega']
        fvelo = sdata['velo']
        floc = sdata['loc']

        fx0 = np.zeros(3)
        fx0[:ndims] = literal_eval(
            soln.config.get('solver-plugin-nirf', 'center-of-rot')
        )

        R = _quat_to_rotmat(fquat)

        # r_body (before rotation) for omega x r in velocity transform
        # Pad to 3D so cross with fomega (always 3D) works in 2D
        r = np.zeros((3, *data.ploc.shape[1:]))
        r[:ndims] = data.ploc - fx0[:ndims, None, None]

        # Transform coordinates: x_lab = R*(x_body - x0) + x0 + loc
        # spts shape is (nspts, neles, ndims) — write in-place so ploc picks it up
        spts = data.spts
        spts -= fx0[:ndims]
        spts[:] = np.einsum('ij,...j->...i', R[:ndims, :ndims], spts)
        spts += fx0[:ndims]

        apply_trans = self.cfg.getbool(self.cfgsect,
                                      'apply-translation', False)
        if apply_trans:
            spts += floc[:ndims]

        # Transform velocity: u_lab = R*(u_body + omega x r_body) + V_frame
        vs = data.pris[1:ndims + 1]
        u_body = np.zeros_like(r)
        u_body[:ndims] = vs

        oxr = np.cross(fomega, r, axisb=0, axisc=0)
        u_lab = np.tensordot(R, u_body + oxr, axes=1) + fvelo[:, None, None]

        # Write back to conservative solution (shape: nupts, nvars, neles)
        # so the downstream export picks up the lab-frame values
        cons = soln.data[data.etype]
        rho = cons[:, 0, :]
        for d in range(ndims):
            du = u_lab[d] - u_body[d]
            cons[:, 1 + d, :] += rho * du
            # Kinetic energy delta: 0.5*rho*(u_lab^2 - u_body^2)
            cons[:, -1, :] += 0.5 * rho * (u_lab[d]**2 - u_body[d]**2)
