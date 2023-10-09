from collections import defaultdict
import math

from pyfr.regions import BoxRegion
import numpy as np

from pyfr.plugins.base import BaseSolverPlugin
from rtree.index import Index, Property


def pcg(seed):
    # pcg32rxs_m_xs
    state = np.uint32(seed)
    multiplier = np.uint32(747796405)
    increment = np.uint32(2891336453)
    mcgmultiplier = np.uint32(277803737)
    n4, n22, n28 = np.uint32(4), np.uint32(22), np.uint32(28)
    with np.errstate(over='ignore'):
        while True:
            ostate = ostatet = state
            state = ostatet*multiplier + increment
            rshift = np.uint32(ostatet >> n28)
            ostatet ^= ostatet >> (n4 + rshift)
            ostatet *= mcgmultiplier
            ostatet ^= ostatet >> n22
            random = (ostatet >> 8) * pow(2, -24)
            yield (ostate, random)


class TurbulencePlugin(BaseSolverPlugin):
    name = 'turbulence'
    systems = ['ac-navier-stokes', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.fdptype = fdptype = intg.backend.fpdtype

        ac = intg.system.name.startswith('ac')

        self.nmvxinit = 6
        self.nvmxtol = 0.1

        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend

        self.seed = self.cfg.getint(cfgsect, 'seed')

        self.eventdtype = [('tinit', fdptype), ('state', np.uint32),
                           ('ts', fdptype), ('te', fdptype)]

        ti = self.cfg.getfloat(cfgsect, 'turbulence-intensity')
        sigma = self.cfg.getfloat(cfgsect, 'sigma')
        self.avgu = avgu = self.cfg.getfloat(cfgsect, 'avg-u')
        self.ls = ls = self.cfg.getfloat(cfgsect, 'turbulence-length-scale')

        gc = (2*sigma/(math.erf(1/sigma)*math.pi**0.5))**0.5
        beta1 = -0.5/(sigma*ls)**2
        beta3 = 0.01*ti*avgu*(gc/sigma)**3

        if not ac:
            gamma = self.cfg.getfloat('constants', 'gamma')
            avgrho = self.cfg.getfloat(cfgsect, 'avg-rho')
            avgmach = self.cfg.getfloat(cfgsect, 'avg-mach')
            beta2 = avgrho*(gamma - 1)*avgmach**2
        else:
            beta2 = 0

        self.ydim = ydim = self.cfg.getfloat(cfgsect, 'y-dim')
        self.zdim = zdim = self.cfg.getfloat(cfgsect, 'z-dim')

        self.ymin = ymin = -ydim/2
        self.zmin = zmin = -zdim/2

        self.nvorts = int(ydim*zdim/(4*ls**2))
        self.bbox = BoxRegion([-2*ls, ymin-ls, zmin-ls],
                              [2*ls, ymin+ydim+ls, zmin+zdim+ls])

        centre = self.cfg.getliteral(cfgsect, 'centre')
        rax = np.array(self.cfg.getliteral(cfgsect, 'rot-axis'))
        ran = np.radians(self.cfg.getfloat(cfgsect, 'rot-angle'))

        self.shift = shift = np.array(centre)
        self.rot = rot = (np.cos(ran)*np.eye(3) +
                          np.sin(ran)*(np.cross(rax, np.eye(3))) +
                          (1 - np.cos(ran))*rax @ rax.T)

        self.macro_params = {'ls': ls, 'avgu': avgu, 'yzmin': [ymin, zmin],
                             'yzdim': [ydim, ydim], 'beta1': beta1,
                             'beta2': beta2, 'beta3': beta3, 'rot': rot,
                             'shift': shift, 'ac': ac}

        self.trcl = defaultdict()
        self.vortbuf = self.getvortbuf()
        self.vortstructs = self.getvortstructs(intg)

        if not bool(self.vortstructs):
            self.tnext = float('inf')

    def __call__(self, intg):
        if intg.tcurr + intg._dt < self.tnext:
            return

        for etype, [strms, strmsid, buf, tinit, state] in self.vortstructs.items():
            if self.trcl[etype] > self.tnext:
                continue

            self.trcl[etype] = self.update_buf(strms, strmsid, buf, self.tnext)
            tinit.set(buf.tinit)
            state.set(buf.state)

        self.tnext = min(self.trcl.values())

    def update_buf(self, strms, strmsid, buf, tnext):
        trcltemp = float('inf')
        for ele, strm in strms.items():
            sid = strmsid[ele]
            if sid >= strm.shape[0]:
                continue

            tmp = buf[:, ele][buf[:, ele].te > tnext]
            rem = tmp.shape[0]
            sft = sid + buf.shape[0] - rem
            add = rem + strm[sid:sft].shape[0]
            buf[:rem, ele] = tmp
            buf[rem:add, ele] = strm[sid:sft]
            buf[add:, ele] = 0

            if sft < strm.shape[0] and (buf[-1, ele].ts < trcltemp):
                trcltemp = buf[-1, ele].ts

            strmsid[ele] = sft

        return trcltemp

    def test_nvmx(self, strms, neles, nvmx):
        tnext = self.tnext
        strmsid = defaultdict()
        for ele in strms:
            strmsid[ele] = 0
        buf = np.core.records.fromrecords(np.zeros((nvmx, neles)),
                                          dtype=self.eventdtype)

        while tnext < float('inf'):
            trcltemp = self.update_buf(strms, strmsid, buf, tnext)
            if trcltemp < (tnext + self.nvmxtol):
                return False

            tnext = trcltemp

        return True

    def getvortbuf(self):
        pcgg = pcg(self.seed)

        vid = 0
        temp = []
        tinits = []

        while vid < self.nvorts:
            tinits.append(self.tstart + 2*self.ls*next(pcgg)[1]/self.avgu)
            vid += 1

        while any(tinit <= self.tend for tinit in tinits):
            for vid, tinit in enumerate(tinits):
                (state, floaty) = next(pcgg)
                yinit = self.ymin + self.ydim*floaty
                zinit = self.zmin + self.zdim*next(pcgg)[1]
                tend = tinit + (2*self.ls/self.avgu)
                if tend >= self.tbegin and tinit <= self.tend:
                    temp.append((yinit, zinit, tinit, state))
                tinits[vid] += 2*self.ls/self.avgu

        return np.core.records.fromrecords(temp,
                                           dtype=[('yinit', self.fdptype),
                                                  ('zinit', self.fdptype),
                                                  ('tinit', self.fdptype),
                                                  ('state', np.uint32)])

    def getvortstructs(self, intg):
        vortstructs = defaultdict()

        for etype, eles in intg.system.ele_map.items():
            neles = eles.neles
            pts = eles.ploc_at_np('upts').swapaxes(0, 1)
            ptsr = (self.rot @ (pts.reshape(3, -1) -
                    self.shift[:, None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)
            inside = self.bbox.pts_in_region(ptsr)

            temp = defaultdict(list)
            strms = defaultdict()
            strmsid = defaultdict()

            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0]
                ptsri = ptsr[:, eids]
                insert = [(i, [*p.min(axis=0), *p.max(axis=0)], None)
                          for i, p in enumerate(ptsri.swapaxes(0, 1))]
                idx3d = Index(insert,
                              properties=Property(dimension=3,
                                                  interleaved=True))

                for vid, vort in enumerate(self.vortbuf):
                    vbmin = [-2*self.ls, vort.yinit - self.ls,
                             vort.zinit - self.ls]
                    vbmax = [2*self.ls, vort.yinit + self.ls,
                             vort.zinit + self.ls]

                    # get candidate points from rtree
                    can = np.array(list(set(idx3d.intersection((*vbmin, *vbmax)))),
                                   dtype=int)

                    # get exact points candidate points
                    vinside = BoxRegion(vbmin, vbmax).pts_in_region(ptsri[:, can])

                    for vi, c in zip(vinside.swapaxes(0, 1), can):
                        if not np.any(vi):
                            continue
                        xv = ptsri[vi, c, 0]
                        xvmin = xv.min()
                        xvmax = xv.max()
                        ts = max(vort.tinit, vort.tinit + xvmin/self.avgu)
                        te = max(ts,
                                 min(ts + (xvmax - xvmin + 2*self.ls)/self.avgu,
                                     vort.tinit + 2*self.ls/self.avgu))

                        # record when a vortex enters and exits an element
                        temp[eids[c]].append((self.vortbuf[vid].tinit,
                                              self.vortbuf[vid].state, ts, te))

                for ele, strm in temp.items():
                    strms[ele] = np.sort(np.core.records.fromrecords(strm,
                                                                     dtype=self.eventdtype),
                                                                     order='ts')
                    strmsid[ele] = 0

                nvmx = self.nmvxinit
                while not self.test_nvmx(strms, neles, nvmx):
                    nvmx += 1

                buf = np.core.records.fromrecords(np.zeros((nvmx, neles)),
                                                  dtype=self.eventdtype)

                vortstruct = [strms, strmsid, buf,
                              intg.backend.matrix((nvmx, neles),
                                                  tags={'align'}),
                              intg.backend.matrix((nvmx, neles),
                                                  tags={'align'},
                                                  dtype=np.uint32)]

                eles.add_src_macro('pyfr.plugins.kernels.turbulence',
                                   'turbulence',
                                   self.macro_params | {'nvmx': nvmx},
                                   ploc=True, soln=True)

                eles._set_external('tinit',
                                   f'in broadcast-col fpdtype_t[{nvmx}]',
                                   value=vortstruct[3])
                       
                eles._set_external('state',
                                   f'in broadcast-col uint32_t[{nvmx}]',
                                   value=vortstruct[4])

                self.trcl[etype] = 0
                vortstructs[etype] = vortstruct

        return vortstructs
