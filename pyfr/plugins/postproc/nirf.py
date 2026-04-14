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
