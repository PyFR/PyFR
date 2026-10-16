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

    def prepare(self, mesh, soln):
        sdata = soln.state.get('plugins/nirf')
        if sdata is None:
            raise RuntimeError('No NIRF state in solution file')

        self._R = _quat_to_rotmat(sdata['quat'])
        self._omega = np.array(sdata['omega'])
        self._velo = np.array(sdata['velo'])
        self._loc = np.array(sdata['loc'])

        # Centre of rotation, padded to 3D
        self._x0 = np.zeros(3)
        self._x0[:self.ndims] = soln.config.getliteral(
            'solver-plugin-nirf', 'center-of-rot', (0.,)*self.ndims
        )

    def _process(self, data):
        ndims = data.ndims
        R, Rd = self._R, self._R[:ndims, :ndims]
        x0 = self._x0[:ndims]
        vnames = ['u', 'v', 'w'][:ndims]

        # Body-frame offsets from the centre of rotation, padded to 3D
        ploc = data.ploc
        r = np.zeros((*ploc.shape[:-1], 3))
        r[..., :ndims] = ploc - x0

        # The wall distance is a frame-invariant scalar, so evaluate it in
        # the body frame before anything moves; the normals rotate
        if self.kind == 'boundary':
            data.boundary_dist
            pnorm, normals = data.pnorm, data.normals
            data.pnorm = np.tensordot(Rd, pnorm, axes=1)
            data.normals = np.tensordot(Rd, normals, axes=1)

        # Transform coordinates: x_lab = R*(x_body - x0) + x0 + loc
        ploc[...] = r[..., :ndims] @ Rd.T + x0
        if self.cfg.getbool(self.cfgsect, 'apply-translation', False):
            ploc += self._loc[:ndims]

        # Transform velocity: u_lab = R*(u_body + omega x r_body) + V_frame
        u = np.zeros_like(r)
        u[..., :ndims] = np.stack([data[n] for n in vnames], axis=-1)
        u = (u + np.cross(self._omega, r)) @ R.T + self._velo
        for i, n in enumerate(vnames):
            data[n][...] = u[..., i]

        if not data.has_grads:
            return

        dims = 'xyz'[:ndims]

        def grad(n):
            return np.stack([data[f'grad_{n}_{d}'] for d in dims], axis=-1)

        def set_grad(n, g):
            for i, d in enumerate(dims):
                data[f'grad_{n}_{d}'][...] = g[..., i]

        # Scalar gradients rotate as covectors: grad_lab = R*grad_body
        privars = self.source.elementscls.privars(ndims, data.cfg)
        for n in privars:
            if n not in vnames:
                set_grad(n, grad(n) @ Rd.T)

        # Velocity gradient tensor: G_lab = R*(G_body + [Omega x])*R^T
        G = np.zeros((*ploc.shape[:-1], 3, 3))
        for i, n in enumerate(vnames):
            G[..., i, :ndims] = grad(n)

        wx, wy, wz = self._omega
        G += np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
        G = R @ G @ R.T
        for i, n in enumerate(vnames):
            set_grad(n, G[..., i, :ndims])
