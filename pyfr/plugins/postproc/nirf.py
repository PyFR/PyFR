import numpy as np

from pyfr.nputil import Quaternion
from pyfr.plugins.postproc.base import BasePostProcPlugin


class NIRFPostProc(BasePostProcPlugin):
    name = 'nirf'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'
    export_types = '.*'
    transform = True
    source_prefix = 'soln'

    def prepare(self, mesh, soln):
        ndims = self.ndims

        if (sdata := soln.state.get('plugins/nirf')) is None:
            raise RuntimeError('No NIRF state in solution file')

        # Form the rotation matrix and the cross product matrix of omega
        self._R = Quaternion(sdata['quat']).to_rotmat()[:ndims, :ndims]
        self._W = np.cross(np.eye(3), sdata['omega'])[:ndims, :ndims]
        self._velo = sdata['velo'][:ndims]

        # Locate the centre of rotation in the body and lab frames
        x0 = soln.config.getliteral('solver-plugin-nirf', 'center-of-rot',
                                    (0,)*ndims)
        self._x0 = np.array(x0, dtype=float)
        self._x0lab = self._x0.copy()
        if self.cfg.getbool(self.cfgsect, 'apply-translation', False):
            self._x0lab += sdata['loc'][:ndims]

    def _process(self, data):
        ndims, R, W = self.ndims, self._R, self._W
        vnames, dims = 'uvw'[:ndims], 'xyz'[:ndims]

        # Rotate the normals into the lab frame
        if self.kind == 'boundary':
            data.normals = np.tensordot(R, data.normals, axes=1)

        ploc = data.ploc
        r = ploc - self._x0

        # Move the points to the lab frame, x_lab = R*(x_body - x0) + x0 + loc
        ploc[:] = r @ R.T + self._x0lab

        # Transform the velocity, u_lab = R*(u_body + omega x r) + V
        u = (data.stack(vnames) + r @ W.T) @ R.T + self._velo
        data.unstack(vnames, u)

        if data.has_grads:
            # Stack the gradients as (..., variable, direction)
            privars = self.source.elementscls.privars(ndims, data.cfg)
            gnames = [f'grad_{n}_{d}' for n in privars for d in dims]
            G = data.stack(gnames).reshape(*ploc.shape[:-1], -1, ndims)

            # Add the frame rotation rate to the velocity gradient, then rotate
            G[..., 1:ndims + 1, :] = R @ (G[..., 1:ndims + 1, :] + W)
            data.unstack(gnames, (G @ R.T).reshape(*ploc.shape[:-1], -1))
