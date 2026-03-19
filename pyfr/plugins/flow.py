from pyfr.plugins.base import BasePostProcPlugin


class MachPostProc(BasePostProcPlugin):
    name = 'mach'
    systems = ['euler', 'navier-stokes']
    dimensions = [2, 3]
    export_types = ['*']

    def fields(self):
        return {'mach': ['Ma']}

    def compute(self, data):
        rho, p = data.pris[0], data.pris[-1]
        vs = data.pris[1:-1]

        gamma = self.cfg.getfloat('constants', 'gamma')

        vmag = sum(v**2 for v in vs)**0.5
        c = (gamma * p / rho)**0.5

        return {'mach': [vmag / c]}


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
