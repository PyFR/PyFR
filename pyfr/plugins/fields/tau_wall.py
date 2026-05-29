import numpy as np

from pyfr.plugins.fields.base import BaseFieldProvider


class TauWallField(BaseFieldProvider):
    name = '_tau_wall'
    systems = 'navier-stokes'
    dimensions = '3'
    export_types = 'boundary'
    needs_grads = True
    deps = ['_mu']

    def _process(self, view):
        normals = view.normals
        mu = view.fields['_mu']

        grad_vel = np.stack(view.grad_pris[1:self.ndims + 1])
        sij = grad_vel + grad_vel.swapaxes(0, 1)
        tau_n = mu * np.einsum('ij...,j...->i...', sij, normals)

        tau_tang = tau_n - (tau_n * normals).sum(axis=0) * normals
        view.fields['_tau_wall'] = np.linalg.norm(tau_tang, axis=0)
