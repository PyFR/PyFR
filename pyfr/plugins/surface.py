import numpy as np

from pyfr.plugins.base import BasePostProcPlugin


class CfPostProc(BasePostProcPlugin):
    name = 'cf'
    systems = ['navier-stokes']
    dimensions = [2, 3]
    export_types = ['boundary']
    needs_gradients = True
    needs_normals = True

    def fields(self):
        return {'cf': ['Cf']}

    def compute(self, data):
        rho_inf = self.cfg.getfloat(self.cfgsect, 'rho-inf')
        u_inf = self.cfg.getfloat(self.cfgsect, 'u-inf')
        mu = self.cfg.getfloat('constants', 'mu')

        q_inf = 0.5 * rho_inf * u_inf**2

        ndims = self.ndims
        normals = data.normals

        # Velocity gradient tensor: grad_vel[i][d] = du_i/dx_d
        grad_vel = np.stack([data.grad_pris[1 + i] for i in range(ndims)])

        # Stress tensor: tau_ij = mu * (du_i/dx_j + du_j/dx_i)
        # Wall shear: tau_w = tau . n - (n . tau . n) * n
        tau_n = mu * np.einsum('ijkl,jkl->ikl',
                               grad_vel + grad_vel.swapaxes(0, 1), normals)

        # Remove normal component
        nn = np.einsum('ikl,ikl->kl', tau_n, normals)
        tau_wall = tau_n - nn[np.newaxis] * normals

        # Cf = |tau_wall| / q_inf
        tau_mag = np.sqrt(np.einsum('ikl,ikl->kl', tau_wall, tau_wall))
        cf = tau_mag / q_inf

        return {'cf': [cf]}


class CpPostProc(BasePostProcPlugin):
    name = 'cp'
    systems = ['euler', 'navier-stokes']
    dimensions = [2, 3]
    export_types = ['boundary']

    def fields(self):
        return {'cp': ['Cp']}

    def compute(self, data):
        p = data.pris[-1]

        rho_inf = self.cfg.getfloat(self.cfgsect, 'rho-inf')
        u_inf = self.cfg.getfloat(self.cfgsect, 'u-inf')
        p_inf = self.cfg.getfloat(self.cfgsect, 'p-inf')

        q_inf = 0.5 * rho_inf * u_inf**2

        return {'cp': [(p - p_inf) / q_inf]}


class YPlusPostProc(BasePostProcPlugin):
    name = 'yplus'
    systems = ['navier-stokes']
    dimensions = [2, 3]
    export_types = ['boundary']
    needs_gradients = True
    needs_normals = True

    def fields(self):
        return {'yplus': ['y+']}

    def compute(self, data):
        mu = self.cfg.getfloat('constants', 'mu')
        ndims = self.ndims
        normals = data.normals
        rho_wall = data.pris[0]

        # Wall shear stress
        grad_vel = np.stack([data.grad_pris[1 + i] for i in range(ndims)])
        tau_n = mu * np.einsum('ijkl,jkl->ikl',
                               grad_vel + grad_vel.swapaxes(0, 1), normals)
        nn = np.einsum('ikl,ikl->kl', tau_n, normals)
        tau_wall = tau_n - nn[np.newaxis] * normals
        tau_mag = np.sqrt(np.einsum('ikl,ikl->kl', tau_wall, tau_wall))

        u_tau = np.sqrt(tau_mag / rho_wall)
        nu = mu / rho_wall
        y = data.wall_dist

        return {'yplus': [y * u_tau / nu]}
