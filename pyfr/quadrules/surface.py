from collections import defaultdict

import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import NoOptionError
from pyfr.quadrules import get_quadrule


class SurfaceIntegrator:
    def _surf_init(self, elemap, surf_list, flags=''):
        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)

        # Element indices and associated face normals
        eidxs = defaultdict(list)
        norms = defaultdict(list)
        locs = defaultdict(list)

        for etype, eidx, fidx in surf_list:
            eles = elemap[etype]
            itype, proj, norm = eles.basis.faces[fidx]

            ppts, pwts = self._surf_quad(itype, proj, flags=flags)
            nppts = len(ppts)

            # Get phyical normals
            ploc = eles.ploc_at_np(ppts)[..., eidx]
            pnorm = eles.pnorm_at(ppts, [norm]*nppts)[:, eidx]

            eidxs[etype, fidx].append(eidx)
            norms[etype, fidx].append(pnorm)
            locs[etype, fidx].append(ploc)

            if (etype, fidx) not in m0:
                m0[etype, fidx] = eles.basis.ubasis.nodal_basis_at(ppts)
                qwts[etype, fidx] = pwts

        self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
        self._norms = {k: np.array(v) for k, v in norms.items()}
        self._locs = {k: np.array(v) for k, v in locs.items()}

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