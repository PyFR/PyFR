import numpy as np

from pyfr.postproc.base import BasePostProcPlugin


class CfPostProc(BasePostProcPlugin):
    name = 'cf'
    systems = ['navier-stokes']
    dimensions = [3]
    export_types = ['boundary']
    needs_gradients = True
    needs_normals = True

    def fields(self):
        return {'cf': ['Cf']}

    def compute(self, data):
        rho_inf = self.cfg.getfloat(self.cfgsect, 'rho-inf')
        u_inf = self.cfg.getfloat(self.cfgsect, 'u-inf')

        q_inf = 0.5 * rho_inf * u_inf**2

        return {'cf': [data.tau_wall / q_inf]}


class YPlusPostProc(BasePostProcPlugin):
    name = 'yplus'
    systems = ['navier-stokes']
    dimensions = [3]
    export_types = ['boundary']
    needs_gradients = True
    needs_normals = True

    def fields(self):
        return {'yplus': ['y+']}

    def compute(self, data):
        mu = self.cfg.getfloat('constants', 'mu')
        rho_wall = data.pris[0]

        u_tau = np.sqrt(data.tau_wall / rho_wall)
        nu = mu / rho_wall

        return {'yplus': [data.wall_dist * u_tau / nu]}
