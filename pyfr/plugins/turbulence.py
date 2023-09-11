import math
import numpy as np

from collections import defaultdict
from pyfr.plugins.base import BaseSolverPlugin
from pyfr.regions import BoxRegion
from rtree.index import Index, Property

def lcg(seed):
    state = np.uint32(seed)
    while True:
        yield state
        with np.errstate(over='ignore'):
            state = state*np.uint32(747796405) + np.uint32(2891336453)

def pcg32rxs_m_xs_random(state):
    with np.errstate(over='ignore'):
        rshift = np.uint32(state >> np.uint32(28))
        state ^= state >> (np.uint32(4) + rshift)
        state *= np.uint32(277803737)
        state ^= state >> np.uint32(22)
        return (state >> 8) * 5.9604644775390625e-8

class TurbulencePlugin(BaseSolverPlugin):
    name = 'turbulence'
    systems = [ 'ac-navier-stokes', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.fdptype = fdptype = intg.backend.fpdtype

        ac = intg.system.name.startswith('ac')

        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend

        self.seed = self.cfg.getint(cfgsect, 'seed')

        self.eventdtype = [('tinit', fdptype), ('state', np.uint32),
                           ('ts', fdptype), ('te', fdptype)]

        gamma = self.cfg.getfloat('constants', 'gamma')
        avgrho = self.cfg.getfloat(cfgsect, 'avg-rho')
        avgmach = self.cfg.getfloat(cfgsect, 'avg-mach')
        ti = self.cfg.getfloat(cfgsect, 'turbulence-intensity')
        sigma = self.cfg.getfloat(cfgsect, 'sigma')
        centre = self.cfg.getliteral(cfgsect, 'centre')
        rax = np.array(self.cfg.getliteral(cfgsect, 'rot-axis'))
        ran = np.radians(self.cfg.getfloat(cfgsect,'rot-angle'))
        self.avgu = avgu = self.cfg.getfloat(cfgsect, 'avg-u')
        self.ls = ls = self.cfg.getfloat(cfgsect, 'turbulence-length-scale')

        gc = (2*sigma/(math.erf(1/sigma)*math.pi**0.5))**0.5
        beta1 = -0.5/(sigma*ls)**2
        beta2 = avgrho*(gamma - 1)*avgmach**2
        beta3 = 0.01*ti*avgu*(gc/sigma)**3

        self.ydim = ydim = self.cfg.getfloat(cfgsect, 'y-dim')
        self.zdim = zdim = self.cfg.getfloat(cfgsect, 'z-dim')
        
        self.ymin = ymin = -ydim/2
        self.zmin = zmin = -zdim/2

        self.nvorts = int(ydim*zdim/(4*ls**2))
        self.bbox = BoxRegion([-2*ls, ymin-ls, zmin-ls],
                              [2*ls, ymin+ydim+ls, zmin+zdim+ls])
        
        self.shift = shift = np.array(centre)
        self.rot = rot = (np.cos(ran)*np.eye(3) -
                          np.sin(ran)*(np.cross(rax, -np.eye(3))) +
                          (1.0 - np.cos(ran))*rax@rax.T)

        self.macro_params = {'ls': ls, 'avgu': avgu, 'yzmin': [ymin, zmin],
                             'yzdim': [ydim, ydim], 'beta1' : beta1, 'beta2': beta2,
                             'beta3': beta3, 'rot': rot, 'shift': shift, 'ac': ac}

        self.vortbuff = self.getvortbuff()
        self.vortstructs = self.getvortstructs(intg)

        if not bool(self.vortstructs):
           self.tnext = float('inf')

    def __call__(self, intg):
        if intg.tcurr + intg._dt < self.tnext:
            return

        for i, [trcl, streams, streamsidx, buff, tinit, state] in enumerate(self.vortstructs):
            if trcl > self.tnext:
                continue

            trcltemp = np.inf
            for ele, stream in streams.items():
                sid = streamsidx[ele]
                if sid >= stream.shape[0]:
                    continue
         
                tmp = buff[:,ele][buff[:,ele].te > self.tnext]
                rem = tmp.shape[0]
                sft = sid + buff.shape[0] - rem
                add = rem + stream[sid:sft].shape[0]
                buff[:rem, ele] = tmp
                buff[rem:add, ele] = stream[sid:sft]
                buff[add:, ele] = 0

                if sft < stream.shape[0] and (buff['ts'][-1, ele] < trcltemp):
                    trcltemp = buff['ts'][-1, ele]

                self.vortstructs[i][2][ele] = sft

            self.vortstructs[i][0] = trcltemp
            tinit.set(buff.tinit)
            state.set(buff.state)
    
        self.tnext = min(etype[0] for etype in self.vortstructs)


    def getvortbuff(self):
        lcgg = lcg(self.seed)

        vid = 0
        temp = []
        tinits = []

        while vid < self.nvorts:
            tinits.append(self.tstart + 2*self.ls*pcg32rxs_m_xs_random(next(lcgg))/self.avgu)
            vid += 1

        while any(tinit <= self.tend for tinit in tinits):     
            for vid, tinit in enumerate(tinits):
                state = next(lcgg)
                yinit = self.ymin + self.ydim*pcg32rxs_m_xs_random(state)
                zinit = self.zmin + self.zdim*pcg32rxs_m_xs_random(next(lcgg))
                if tinit+(2*self.ls/self.avgu) >= self.tbegin and tinit <= self.tend:
                    temp.append((yinit, zinit, tinit, state))
                tinits[vid] += 2*self.ls/self.avgu
    
        return np.core.records.fromrecords(temp, dtype = [('yinit', self.fdptype),
                                                          ('zinit', self.fdptype),
                                                          ('tinit', self.fdptype),
                                                          ('state', np.uint32)])

    def getvortstructs(self, intg):
        vortstructs = []
        for etype, eles in intg.system.ele_map.items(): 
            neles = eles.neles
            pts = eles.ploc_at_np('upts').swapaxes(0, 1)
            ptsr = (self.rot @ (pts.reshape(3, -1) - self.shift[:,None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)
            inside = self.bbox.pts_in_region(ptsr)

            temp = defaultdict(list)
            streams = defaultdict()
            streamsidx = defaultdict()

            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0]
                ptsri = ptsr[:,eids]
                insert = [(i, [*p.min(axis=0), *p.max(axis=0)], None)
                          for i, p in enumerate(ptsri.swapaxes(0, 1))]
                idx3d = Index(insert, properties=Property(dimension=3, interleaved=True))
                
                for vid, vort in enumerate(self.vortbuff):
                    elestemp = []
                    vbmin = [-2*self.ls, vort.yinit - self.ls, vort.zinit - self.ls]
                    vbmax = [2*self.ls, vort.yinit + self.ls, vort.zinit + self.ls]

                    # get candidate points from rtree
                    can = np.array(list(set(idx3d.intersection((*vbmin, *vbmax)))), dtype=int)   
                    
                    # get exact points candidate points
                    vinside = BoxRegion(vbmin, vbmax).pts_in_region(ptsri[:,can])

                    for vi, c in zip(vinside.swapaxes(0, 1), can):
                        if not np.any(vi):
                            continue
                        xv = ptsri[vi, c, 0]
                        xvmin = xv.min()
                        xvmax = xv.max()
                        ts = max(vort.tinit, vort.tinit + xvmin/self.avgu)
                        te = max(ts, min(ts + (xvmax-xvmin+2*self.ls)/self.avgu, vort.tinit + 2*self.ls/self.avgu))

                        # record when a vortex enters and exits an element
                        temp[eids[c]].append((self.vortbuff[vid].tinit, self.vortbuff[vid].state, ts, te))

                for ele, stream in temp.items():
                    streams[ele] = np.sort(np.core.records.fromrecords(stream, dtype = self.eventdtype), order='ts')
                    streamsidx[ele] = 0

                nvmx = 0
                for stream in streams.values():
                    for i, ts in enumerate(stream.ts):
                        cnt = 0
                        while i-cnt >= 0:
                            if stream.te[i-cnt] < ts:
                                break
                            cnt += 1
                        nvmx = max(nvmx, cnt)

                nvmx += 1

                buff = np.core.records.fromrecords(np.zeros((nvmx, neles)), dtype = self.eventdtype)

                vortstruct = [0.0, streams, streamsidx, buff, intg.backend.matrix((nvmx, neles), tags={'align'}), intg.backend.matrix((nvmx, neles), tags={'align'}, dtype=np.uint32)]  

                eles.add_src_macro('pyfr.plugins.kernels.turbulence',
                                   'turbulence',
                                   self.macro_params | {'nvmx': nvmx},
                                   ploc=True, soln=True)

                eles._set_external('tinit',
                                   f'in broadcast-col fpdtype_t[{nvmx}]',
                                   value=vortstruct[4])
                                   
                eles._set_external('state',
                                   f'in broadcast-col uint32_t[{nvmx}]',
                                   value=vortstruct[5])

                vortstructs.append(vortstruct)

        return vortstructs
