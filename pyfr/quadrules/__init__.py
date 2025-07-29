from collections import defaultdict
from importlib.resources import files
import re

import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import NoOptionError


class BaseTabulatedQuadRule:
    def __init__(self, rule, flags=None):
        pts = []
        wts = []

        rule = re.sub(r'(?<=\))\s*,?\s*(?!$)', r'\n', rule)
        rule = re.sub(r'\(|\)|,', '', rule).strip()
        rule = rule[1:-1] if rule.startswith('[') else rule

        for l in rule.splitlines():
            if not l:
                continue

            # Parse the line
            args = [float(f) for f in l.split()]

            if len(args) == self.ndim:
                pts.append(args)
            elif len(args) == self.ndim + 1:
                pts.append(args[:-1])
                wts.append(args[-1])
            else:
                raise ValueError('Invalid points in quadrature rule')

        if len(wts) and len(wts) != len(pts):
            raise ValueError('Invalid number of weights')

        # Flatten 1D rules
        if self.ndim == 1:
            pts = [p[0] for p in pts]

        # Cast and assign
        self.pts = np.array(pts)
        self.wts = np.array(wts)
        self.flags = frozenset(flags or '')


class BaseStoredQuadRule(BaseTabulatedQuadRule):
    @classmethod
    def _iter_rules(cls):
        if not hasattr(cls, '_rpaths'):
            cls._rpaths = list(files(__name__).joinpath(cls.shape).iterdir())

        for path in cls._rpaths:
            m = re.match(r'([a-zA-Z0-9\-~+]+)-n(\d+)'
                         r'(?:-d(\d+))?(?:-([pstu]+))?\.txt$', path.name)
            if m:
                yield (path, m[1], int(m[2]), int(m[3] or -1), set(m[4] or ''))

    def __init__(self, name=None, npts=None, qdeg=None, flags=None):
        if not npts and not qdeg:
            raise ValueError('Must specify either npts or qdeg')

        best = None
        for rpath, rname, rnpts, rqdeg, rflags in self._iter_rules():
            # See if this rule fulfils the required criterion
            if ((not name or name == rname) and
                (not npts or npts == rnpts) and
                (not qdeg or qdeg <= rqdeg) and
                (not flags or set(flags) <= rflags)):
                # If so see if it is better than the current candidate
                if (not best or
                    (npts and rqdeg > best[2]) or
                    (qdeg and rnpts < best[1])):
                    best = (rpath, rnpts, rqdeg, rflags)

        # Raise if no suitable rules were found
        if not best:
            raise ValueError('No suitable quadrature rule found')

        # Load the rule
        super().__init__(best[0].read_text(), rflags)


def get_quadrule(eletype, rule=None, npts=None, qdeg=None, flags=None):
    ndims = dict(line=1, quad=2, tri=2, hex=3, pri=3, pyr=3, tet=3)

    if rule and not re.match(r'[a-zA-z0-9\-~+]+$', rule):
        class TabulatedQuadRule(BaseTabulatedQuadRule):
            shape = eletype
            ndim = ndims[eletype]

        r = TabulatedQuadRule(rule)

        # Validate the provided point set
        if npts and npts != len(r.pts):
            raise ValueError('Invalid number of points in provided rule')

        if qdeg and not len(r.wts):
            raise ValueError('Provided rule has no quadrature weights')

        return r
    else:
        class StoredQuadRule(BaseStoredQuadRule):
            shape = eletype
            ndim = ndims[eletype]

        return StoredQuadRule(rule, npts, qdeg, flags)


class SurfaceMixin:
    def _surf_init(self, elemap, surf_list, quad_flags=''):
        # Interpolation matrices and quadrature weights
        self._m0 = m0 = {}
        self._qwts = qwts = defaultdict(list)
        self._m4 = m4 = {}
        rcpjact = {}

        # Element indices and associated face normals
        eidxs = defaultdict(list)
        norms = defaultdict(list)
        locs = defaultdict(list)

        for etype, eidx, fidx in surf_list:
            eles = elemap[etype]
            itype, proj, norm = eles.basis.faces[fidx]

            ppts, pwts = self._surf_quad(itype, proj, flags=quad_flags)
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
            
            if etype not in m4:
                m4[etype] = eles.basis.m4

                # Get the smats at the solution points
                smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                # Get |J|^-1 at the solution points
                rcpdjac = eles.rcpdjac_at_np('upts')

                # Product to give J^-T at the solution points
                rcpjact[etype] = smat*rcpdjac

        self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
        self._norms = {k: np.array(v) for k, v in norms.items()}
        self._locs = {k: np.array(v) for k, v in locs.items()}
        self._rcpjact = {k: rcpjact[k[0]][..., v] 
                        for k, v in self._eidxs.items()}

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