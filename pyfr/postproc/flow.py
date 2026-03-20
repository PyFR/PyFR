import numpy as np

from pyfr.postproc.base import BasePostProcPlugin


class MachPostProc(BasePostProcPlugin):
    name = 'mach'
    systems = ['euler', 'navier-stokes']
    dimensions = [2, 3]
    export_types = ['*']

    def fields(self):
        return {'mach': ['Ma']}

    def compute(self, data):
        rho, *vs, p = data.pris

        gamma = self.cfg.getfloat('constants', 'gamma')
        vmag = np.sqrt(sum(v**2 for v in vs))
        c = np.sqrt(gamma * p / rho)

        return {'mach': [vmag / c]}


class CpPostProc(BasePostProcPlugin):
    name = 'cp'
    systems = ['euler', 'navier-stokes']
    dimensions = [2, 3]
    export_types = ['*']

    def fields(self):
        return {'cp': ['Cp']}

    def compute(self, data):
        p = data.pris[-1]

        rho_inf = self.cfg.getfloat(self.cfgsect, 'rho-inf')
        u_inf = self.cfg.getfloat(self.cfgsect, 'u-inf')
        p_inf = self.cfg.getfloat(self.cfgsect, 'p-inf')

        q_inf = 0.5 * rho_inf * u_inf**2

        return {'cp': [(p - p_inf) / q_inf]}


class VorticityPostProc(BasePostProcPlugin):
    name = 'vorticity'
    systems = ['*']
    dimensions = [2, 3]
    export_types = ['*']
    needs_gradients = True

    def fields(self):
        if self.ndims == 3:
            return {'vorticity': ['omega_x', 'omega_y', 'omega_z']}
        else:
            return {'vorticity': ['omega_z']}

    def compute(self, data):
        du = data.grad_pris[1]
        dv = data.grad_pris[2]

        if self.ndims == 3:
            dw = data.grad_pris[3]

            omega_x = dw[1] - dv[2]
            omega_y = du[2] - dw[0]
            omega_z = dv[0] - du[1]

            return {'vorticity': [omega_x, omega_y, omega_z]}
        else:
            omega_z = dv[0] - du[1]

            return {'vorticity': [omega_z]}
