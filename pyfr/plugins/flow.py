from pyfr.plugins.base import BasePostProcPlugin


class MachPostProc(BasePostProcPlugin):
    name = 'mach'
    systems = ['euler', 'navier-stokes']
    dimensions = [2, 3]
    export_types = ['*']

    def fields(self):
        return {'mach': ['Ma']}

    def compute(self, pris, grad_pris, ploc):
        rho, p = pris[0], pris[-1]
        vs = pris[1:-1]

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

    def compute(self, pris, grad_pris, ploc):
        # grad_pris[i] shape: (ndims, npts, neles)
        # velocity is at indices 1, 2, (3) for all systems
        du = grad_pris[1]
        dv = grad_pris[2]

        if self.ndims == 3:
            dw = grad_pris[3]

            omega_x = dw[1] - dv[2]  # dw/dy - dv/dz
            omega_y = du[2] - dw[0]  # du/dz - dw/dx
            omega_z = dv[0] - du[1]  # dv/dx - du/dy

            return {'vorticity': [omega_x, omega_y, omega_z]}
        else:
            omega_z = dv[0] - du[1]  # dv/dx - du/dy

            return {'vorticity': [omega_z]}
