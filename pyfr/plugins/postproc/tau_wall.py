import numpy as np

from pyfr.plugins.postproc.base import BasePostProcPlugin


class TauWallPostProc(BasePostProcPlugin):
    name = '_tau_wall'
    systems = 'navier-stokes'
    dimensions = '3'
    export_types = 'boundary'
    needs_grads = True

    def fields(self):
        return {}

    def _process(self, data):
        normals = data.normals

        grad_vel = np.stack(data.grad_pris[1:data.ndims + 1])
        sij = grad_vel + grad_vel.swapaxes(0, 1)
        tau_n = data.mu * np.einsum('ijkl,jkl->ikl', sij, normals)

        tau_tang = tau_n - (tau_n * normals).sum(axis=0) * normals
        data.fields['_tau_wall'] = np.linalg.norm(tau_tang, axis=0)
