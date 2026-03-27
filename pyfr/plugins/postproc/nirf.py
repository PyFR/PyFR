from ast import literal_eval

import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin
from pyfr.plugins.solver.nirf import _quat_to_rotmat


class NIRFPostProc(BasePostProcPlugin):
    name = 'nirf'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'

    def process(self, data):
        ndims = data.ndims
        cfg = data.soln.config
        sect = 'solver-plugin-nirf'

        sdata = data.soln.state.get('plugins/nirf')
        if sdata is None:
            return

        fquat = sdata['quat']
        fomega = sdata['omega']
        fvelo = sdata['velo']
        floc = sdata['loc']

        fx0 = np.zeros(3)
        fx0[:ndims] = literal_eval(cfg.get(sect, 'center-of-rot'))

        R = _quat_to_rotmat(fquat)

        # Transform coordinates: x_lab = R*(x_body - x0) + x0 + loc
        ploc = data.ploc
        for d in range(3):
            ploc[d] -= fx0[d]

        r = np.array(ploc)

        rloc = np.einsum('ij,j...->i...', R, ploc)
        for d in range(3):
            ploc[d] = rloc[d] + fx0[d] + floc[d]

        # Transform velocity: u_lab = R*(u_body + omega x r_body) + V_frame
        vs = data.pris[1:ndims + 1]

        # omega x r (always use full 3D cross product)
        oxr = np.array([fomega[1]*r[2] - fomega[2]*r[1],
                        fomega[2]*r[0] - fomega[0]*r[2],
                        fomega[0]*r[1] - fomega[1]*r[0]])

        for d in range(ndims):
            vs[d][:] += oxr[d]

        u_rot = np.array(vs)
        for d in range(ndims):
            vs[d][:] = sum(R[d, k]*u_rot[k] for k in range(ndims)) + fvelo[d]
