import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import NoOptionError
from pyfr.quadrules import get_quadrule


class SurfaceIntegrator:
    def __init__(self, cfg, cfgsect, elemap, con, flags=''):
        self.cfg, self.cfgsect = cfg, cfgsect

        self.m0, self.qwts = {}, {}
        self.eidxs, self.norms, self.locs = {}, {}, {}

        if con is None:
            return

        for etype, fidx, eidxs in con.items():
            eles = elemap[etype]
            itype, proj, norm = eles.basis.faces[fidx]
            ppts, pwts = self._surf_quad(itype, proj, flags=flags)
            nppts = len(ppts)

            ploc = eles.ploc_at_np(ppts)[..., eidxs]
            # HACK: swapaxes so pnorm matches ploc layout (nfpts, ndims, neles)
            pnorm = eles.pnorm_at(ppts, [norm]*nppts)[:, eidxs].swapaxes(1, 2)

            self.m0[etype, fidx] = eles.basis.ubasis.nodal_basis_at(ppts)
            self.qwts[etype, fidx] = pwts
            self.eidxs[etype, fidx] = eidxs
            self.norms[etype, fidx] = pnorm.transpose(2, 0, 1)
            self.locs[etype, fidx] = ploc.transpose(1, 0, 2)

    @memoize
    def _surf_quad(self, itype, proj, flags=''):
        # Obtain quadrature info
        rname = self.cfg.get(f'solver-interfaces-{itype}', 'flux-pts')

        # Quadrature rule (default to that of the solution points)
        qrule = self.cfg.get(self.cfgsect, f'quad-pts-{itype}', rname)
        try:
            qdeg = self.cfg.getint(self.cfgsect, f'quad-deg-{itype}')
        except NoOptionError:
            qdeg = self.cfg.getint(self.cfgsect, 'quad-deg')

        # Get the quadrature rule
        q = get_quadrule(itype, qrule, qdeg=qdeg, flags=flags)

        # Project its points onto the provided surface
        pts = np.atleast_2d(q.pts.T)

        return np.vstack(np.broadcast_arrays(*proj(*pts))).T, q.wts
