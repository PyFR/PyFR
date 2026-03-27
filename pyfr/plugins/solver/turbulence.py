from dataclasses import dataclass
import math

import numpy as np
from numpy.core.records import fromrecords
from rtree.index import Index, Property

from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.regions import BoxRegion


@dataclass
class VortexData:
    strms: dict
    strmoff: dict
    buf: object
    tinit: object
    state: object
    trefill: float = 0.0


def pcg32rxs_m_xs(seed):
    state = np.uint32(seed)
    multiplier = np.uint32(747796405)
    increment = np.uint32(2891336453)
    mcgmultiplier = np.uint32(277803737)
    n4, n22, n28 = np.uint32(4), np.uint32(22), np.uint32(28)
    with np.errstate(over='ignore'):
        while True:
            ostate = ostatet = state
            state = ostatet*multiplier + increment
            rshift = ostatet >> n28
            ostatet ^= ostatet >> (n4 + rshift)
            ostatet *= mcgmultiplier
            ostatet ^= ostatet >> n22
            random = (ostatet >> 8) * 2**-24
            yield (ostate, random)


class TurbulencePlugin(BaseSolverPlugin):
    name = 'turbulence'
    systems = ['navier-stokes']
    dimensions = [3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.fdptype = fdptype = intg.backend.fpdtype

        self.nvmaxtol = 0.1

        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend

        self.seed = 42

        self.eventdtype = [('tinit', fdptype), ('state', np.uint32),
                           ('ts', fdptype), ('te', fdptype)]

        ti = self.cfg.getfloat(cfgsect, 'turbulence-intensity')
        sigma = self.cfg.getfloat(cfgsect, 'sigma')
        self.avgu = avgu = self.cfg.getfloat(cfgsect, 'avg-u')
        self.ls = ls = self.cfg.getfloat(cfgsect, 'turbulence-length-scale')

        gc = (2*sigma/(math.erf(1/sigma)*math.pi**0.5))**0.5
        beta1 = -0.5/(sigma*ls)**2
        beta3 = 0.01*ti*avgu*(gc/sigma)**3

        gamma = self.cfg.getfloat('constants', 'gamma')
        avgrho = self.cfg.getfloat(cfgsect, 'avg-rho')
        avgmach = self.cfg.getfloat(cfgsect, 'avg-mach')
        beta2 = (avgrho/avgu)*(gamma - 1)*avgmach**2

        ydim = self.cfg.getfloat(cfgsect, 'y-dim')
        zdim = self.cfg.getfloat(cfgsect, 'z-dim')

        self.nvorts = int(ydim*zdim / (4*ls**2))
        self.yzdim = yzdim = np.array([ydim, zdim])
        self.bbox = BoxRegion([-2*ls, *(-0.5*yzdim - ls)],
                              [2*ls, *(0.5*yzdim + ls)])

        centre = self.cfg.getliteral(cfgsect, 'centre')
        rax = np.array(self.cfg.getliteral(cfgsect, 'rot-axis'))
        ran = np.radians(self.cfg.getfloat(cfgsect, 'rot-angle'))

        self.shift = shift = np.array(centre)
        self.rot = rot = (np.cos(ran)*np.eye(3) +
                          np.sin(ran)*(np.cross(rax, np.eye(3))) +
                          (1 - np.cos(ran))*np.outer(rax, rax))

        self.macro_params = {
            'ls': ls, 'avgu': avgu, 'yzdim': yzdim, 'beta1': beta1,
            'beta2': beta2, 'beta3': beta3, 'rot': rot, 'shift': shift
        }

        self.vortexbuf = self._get_vortex_buf()
        self.vortexdata = self._get_vortex_data(intg)

        if not self.vortexdata:
            self.tnext = float('inf')

    def __call__(self, intg):
        if intg.tcurr + intg.dt < self.tnext:
            return

        for vs in self.vortexdata.values():
            if vs.trefill > self.tnext:
                continue

            vs.trefill = self._update_buf(vs.strms, vs.strmoff, vs.buf,
                                          self.tnext)
            vs.tinit.set(vs.buf.tinit)
            vs.state.set(vs.buf.state)

        self.tnext = min(vs.trefill for vs in self.vortexdata.values())

    def _update_buf(self, strms, strmoff, buf, tnext):
        trefill = float('inf')
        for ele, strm in strms.items():
            sid = strmoff[ele]
            if sid >= len(strm):
                continue

            # Compact: keep only events that have not yet expired
            tmp = buf[:, ele][buf[:, ele].te > tnext]
            rem = len(tmp)

            # Fill remaining buffer slots from the stream
            new = strm[sid:sid + len(buf) - rem]
            sft, add = sid + len(new), rem + len(new)
            buf[:add, ele] = np.concatenate([tmp, new])
            buf[add:, ele] = 0

            # Once the last buffered event starts all slots are active
            if sft < len(strm) and buf[-1, ele].ts < trefill:
                trefill = buf[-1, ele].ts

            strmoff[ele] = sft

        return trefill

    def _test_nvmax(self, strms, neles, nvmax):
        tnext = self.tnext
        strmoff = {ele: 0 for ele in strms}
        buf = fromrecords(np.zeros((nvmax, neles)), dtype=self.eventdtype)

        while tnext < float('inf'):
            trcltmp = self._update_buf(strms, strmoff, buf, tnext)
            if trcltmp < tnext + self.nvmaxtol:
                return False

            tnext = trcltmp

        return True

    def _find_nvmax(self, strms, neles):
        # Obtain a lower bound on the number of buffer slots needed
        lo = 1
        for strm in strms.values():
            # Interleave start (+1) and end (-1) times for each vortex
            ev = np.stack([strm['ts'], strm['te']], axis=1).ravel()
            sg = np.tile([1, -1], len(strm))

            # Compute the maximum simultaneous overlap
            lo = max(lo, np.cumsum(sg[np.argsort(ev, kind='stable')]).max())

        # Linear search from this lower bound
        while not self._test_nvmax(strms, neles, lo):
            lo += 1

        # Add one to reduce update frequency
        return lo + 1

    def _get_vortex_buf(self):
        dt = 2*self.ls/self.avgu
        pcg = pcg32rxs_m_xs(self.seed)

        tmp = []
        tinits = [self.tstart + dt*next(pcg)[1] for _ in range(self.nvorts)]

        while any(tinit <= self.tend for tinit in tinits):
            for vid, tinit in enumerate(tinits):
                state, floaty = next(pcg)
                yinit = self.yzdim[0]*(-0.5 + floaty)
                zinit = self.yzdim[1]*(-0.5 + next(pcg)[1])
                tend = tinit + dt
                if tend >= self.tbegin and tinit <= self.tend:
                    tmp.append((yinit, zinit, tinit, state))

                tinits[vid] += dt

        dtype = [('yinit', self.fdptype), ('zinit', self.fdptype),
                 ('tinit', self.fdptype), ('state', np.uint32)]
        return fromrecords(tmp, dtype=dtype)

    def _get_vortex_data(self, intg):
        ls, avgu = self.ls, self.avgu
        props = Property(dimension=3)
        eventdtype = self.eventdtype
        data = {}

        for etype, eles in intg.system.ele_map.items():
            # Transform solution points into the plugin's frame
            pts = eles.ploc_at_np('upts').swapaxes(0, 1)
            ptsr = (self.rot @ (pts.reshape(3, -1) -
                    self.shift[:, None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)

            # Skip element types with no points in the inflow region
            inside = self.bbox.pts_in_region(ptsr)
            if not np.any(inside):
                continue

            strms = {}

            # Restrict to elements with at least one point inside
            eids = np.flatnonzero(np.any(inside, axis=0))
            ptsri = ptsr[:, eids]
            bmins, bmaxs = ptsri.min(axis=0), ptsri.max(axis=0)
            rtree = Index((np.arange(len(eids)), bmins, bmaxs),
                          properties=props)

            # Find (vortex, element) pairs whose bounding boxes overlap
            nvort = len(self.vortexbuf)
            vy, vz = self.vortexbuf.yinit, self.vortexbuf.zinit
            vmins = np.column_stack([np.full(nvort, -2*ls), vy - ls, vz - ls])
            vmaxs = np.column_stack([np.full(nvort, 2*ls), vy + ls, vz + ls])
            ids, cnts = rtree.intersection_v(vmins, vmaxs)
            vids = np.repeat(np.arange(nvort), cnts.astype(int))

            # Refine to solution points actually inside each vortex box
            pp = ptsri[:, ids]
            bmin, bmax = vmins[vids][None], vmaxs[vids][None]
            in_box = np.all((pp >= bmin) & (pp <= bmax), axis=-1)

            # x-extent of interior points determines convection window
            ok = np.any(in_box, axis=0)
            xm = np.where(in_box, pp[:, :, 0], np.nan)
            xvmin = np.nanmin(xm[:, ok], axis=0)
            xvmax = np.nanmax(xm[:, ok], axis=0)

            # Compute when each vortex enters and exits each element
            vb = self.vortexbuf[vids[ok]]
            tinit, state = vb.tinit, vb.state
            ts = np.maximum(tinit, tinit + xvmin / avgu)
            tend = np.minimum(ts + (xvmax - xvmin + 2*ls) / avgu,
                              tinit + 2*ls / avgu)
            te = np.maximum(ts, tend)

            # Group events by element into sorted stream arrays
            elarr = eids[ids[ok]]
            order = np.argsort(elarr, kind='stable')
            elarr, tinit, state, ts, te = (a[order] for a in
                                           [elarr, tinit, state, ts, te])
            ueles, starts = np.unique(elarr, return_index=True)
            ends = np.append(starts[1:], len(elarr))
            for ele, s, e in zip(ueles, starts, ends):
                rec = np.empty(e - s, dtype=eventdtype)
                rec['tinit'] = tinit[s:e]
                rec['state'] = state[s:e]
                rec['ts'] = ts[s:e]
                rec['te'] = te[s:e]
                strms[ele] = np.sort(rec, order='ts')

            # Register kernel, allocate buffers, and wire up externs
            data[etype] = self._commit_vortex_data(intg.backend, eles, strms)

        return data

    def _commit_vortex_data(self, backend, eles, strms):
        neles = eles.neles

        # Find the minimum buffer size that avoids overflow
        nvmax = self._find_nvmax(strms, neles)

        # Register the turbulence source macro for this element type
        eles.add_src_macro(
            'pyfr.plugins.kernels.turbulence', 'turbulence',
            self.macro_params | {'nvmax': nvmax}, ploc=True, soln=True
        )

        # Allocate host buffer and backend matrices for vortex state
        buf = fromrecords(np.zeros((nvmax, neles)), dtype=self.eventdtype)

        strmoff = {ele: 0 for ele in strms}
        vdata = VortexData(
            strms, strmoff, buf,
            backend.matrix((nvmax, neles), tags={'align'}),
            backend.matrix((nvmax, neles), tags={'align'}, dtype=np.uint32)
        )

        # Wire up tinit and state as broadcast-column externals
        eles._set_external('tinit', f'in broadcast-col fpdtype_t[{nvmax}]',
                           value=vdata.tinit)
        eles._set_external('state', f'in broadcast-col uint32_t[{nvmax}]',
                           value=vdata.state)

        return vdata
