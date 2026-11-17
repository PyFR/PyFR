import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin
from pyfr.plugins.solver.nirf import quat_to_rotmat


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

        # Rotation and angular velocity cross product matrices
        self._R = quat_to_rotmat(sdata['quat'])[:ndims, :ndims]
        self._W = np.cross(np.eye(3), sdata['omega'])[:ndims, :ndims]
        self._velo = sdata['velo'][:ndims]

        # Centre of rotation in the body and lab frames
        x0 = soln.config.getliteral('solver-plugin-nirf', 'center-of-rot',
                                    (0.,)*ndims)
        self._x0 = np.array(x0, dtype=float)
        self._x0lab = self._x0.copy()
        if self.cfg.getbool(self.cfgsect, 'apply-translation', False):
            self._x0lab += sdata['loc'][:ndims]

    def _process(self, data):
        ndims, R, W = self.ndims, self._R, self._W
        vnames, dims = 'uvw'[:ndims], 'xyz'[:ndims]

        # Evaluate the wall distance before the points and normals rotate
        if self.kind == 'boundary':
            data.boundary_dist
            data.normals = np.tensordot(R, data.normals, axes=1)

        # Body-frame offsets from the centre of rotation
        ploc = data.ploc
        r = ploc - self._x0

        # Lab-frame position, x_lab = R*(x_body - x0) + x0 + loc
        ploc[...] = r @ R.T + self._x0lab

        # Lab-frame velocity, u_lab = R*(u_body + omega x r) + V
        u = np.stack([data[n] for n in vnames], axis=-1)
        u = (u + r @ W.T) @ R.T + self._velo
        for i, n in enumerate(vnames):
            data[n][...] = u[..., i]

        if not data.has_grads:
            return

        # Gradients as (..., variable, direction)
        privars = self.source.elementscls.privars(ndims, data.cfg)
        gnames = [[f'grad_{n}_{d}' for d in dims] for n in privars]
        G = np.stack([np.stack([data[k] for k in row], axis=-1)
                      for row in gnames], axis=-2)

        # The velocity gradient picks up the frame rotation rate
        G[..., 1:ndims + 1, :] = R @ (G[..., 1:ndims + 1, :] + W)
        G = G @ R.T
        for i, row in enumerate(gnames):
            for j, k in enumerate(row):
                data[k][...] = G[..., i, j]
