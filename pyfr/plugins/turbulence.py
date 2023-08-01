import math
import numpy as np

from collections import defaultdict
from pyfr.plugins.base import BaseSolverPlugin
from pyfr.regions import BoxRegion
from rtree.index import Index, Property


class pcg32rxs_m_xs:
    def __init__(self, seed):
        self.state = np.uint32(seed)
    def random(self):
        oldstate = np.uint32(self.state)
        self.state = (oldstate * np.uint32(747796405)) + np.uint32(2891336453)
        rshift = np.uint32(oldstate >> np.uint32(28))
        oldstate ^= oldstate >> (np.uint32(4) + rshift)
        oldstate *= np.uint32(277803737)
        oldstate ^= oldstate >> np.uint32(22)
        return (oldstate >> 8) * 5.9604644775390625e-8
    def getstate(self):
        return self.state


class TurbulencePlugin(BaseSolverPlugin):
    name = 'turbulence'
    systems = ['navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [3]

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        fdptype = intg.backend.fpdtype

        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend

        self.seed = self.cfg.getint(cfgsect, 'seed')

        self.vortbuffdtype = np.dtype([('yinit', fdptype), ('zinit', fdptype),
                                   ('tinit', fdptype), ('state', np.uint32)])
        self.streamdtype = np.dtype([('vid', '<i4'), ('ts', fdptype),
                                          ('te', fdptype)])
        self.eventbuffdtype = np.dtype([('tinit', fdptype), ('state', np.uint32),
                                    ('ts', fdptype), ('te', fdptype)])

        gamma = self.cfg.getfloat('constants', 'gamma')
        avgrho = self.cfg.getfloat(cfgsect, 'avg-rho')
        avgmach = self.cfg.getfloat(cfgsect, 'avg-mach')
        ti = self.cfg.getfloat(cfgsect, 'turbulence-intensity')
        sigma = self.cfg.getfloat(cfgsect, 'sigma')
        centre = self.cfg.getliteral(cfgsect, 'centre')
        rotaxis = np.array(self.cfg.getliteral(cfgsect, 'rot-axis'))
        rotangle = -1.0*np.radians(self.cfg.getfloat(cfgsect,'rot-angle'))
        self.avgu = avgu = self.cfg.getfloat(cfgsect, 'avg-u')
        self.ls = ls = self.cfg.getfloat(cfgsect, 'turbulence-length-scale')

        gc = (2*sigma/(math.erf(1/sigma)*math.pi**0.5))**0.5
        fac1 = -0.5/(sigma*ls)**2
        fac2 = avgrho*(gamma - 1)*avgmach**2
        fac3 = 0.01*ti*avgu*(gc/sigma)**3

        self.ydim = ydim = self.cfg.getfloat(cfgsect, 'y-dim')
        self.zdim = zdim = self.cfg.getfloat(cfgsect, 'z-dim')
        
        self.ymin = ymin = -ydim/2
        self.zmin = zmin = -zdim/2

        self.nvorts = int(ydim*zdim/(4*ls**2))
        self.bbox = BoxRegion([-2*ls, ymin-ls, zmin-ls],
                              [2*ls, ymin+ydim+ls, zmin+zdim+ls])
        
        self.shift = shift = np.array(centre)
        self.rot = rot = np.cos(rotangle)*np.identity(3) +\
                         np.sin(rotangle)*(np.cross(rotaxis, np.identity(3) * -1)) +\
                         (1.0-np.cos(rotangle))*np.outer(rotaxis,rotaxis)

        if hasattr(intg, 'dtmax'):
            self.dtol = intg.dtmax
        else:
            self.dtol = intg._dt

        self.macro_params = {'ls': ls, 'avgu': avgu, 'yzmin': [ymin, zmin],
                             'yzdim': [ydim, ydim], 'fac1' : fac1, 'fac2': fac2,
                             'fac3': fac3, 'rot': rot, 'shift': shift}

        self.vortbuff = self.getvortbuff()
        self.vortstructs = self.getvortstructs(intg)

        if not bool(self.vortstructs):
           self.tnext = float('inf')

    def __call__(self, intg):
        tcurr = intg.tcurr
        if tcurr + self.dtol < self.tnext:
            return

        for i, vortstruct in enumerate(self.vortstructs):
            if vortstruct['trcl'] <= self.tnext:
                trcl = np.inf
                for j, stream in vortstruct['stream'].items():
                    if stream['vid'].any():
                        temp1 = vortstruct['eventbuff'][:,j][vortstruct['eventbuff'][:,j]['te'] > tcurr]       
                        shift = vortstruct['eventbuff'].shape[0]-len(temp1)   
                        if shift:
                            temp2 = self.vortbuff[['tinit', 'state']][stream['vid'][:shift]]
                            pad = shift-temp2.shape[0]
                            temp3 = np.zeros(shift, self.eventbuffdtype)
                            temp3[['tinit', 'state']] = np.pad(temp2, (0,pad), 'constant')
                            temp3[['ts', 'te']] = np.pad(stream[['ts', 'te']][:shift], (0,pad), 'constant')
                            self.vortstructs[i]['eventbuff'][:,j] = np.concatenate((temp1,temp3))
                            self.vortstructs[i]['stream'][j] = stream[shift:]
                        else:
                            if tcurr >= stream['ts'][0]:
                                raise RuntimeError('Active vortex in stream cannot fit on buffer.')
                            
                        if self.vortstructs[i]['stream'][j]['vid'].any() and (self.vortstructs[i]['eventbuff'][-1,j]['ts'] < trcl):
                            trcl = self.vortstructs[i]['eventbuff'][-1,j]['ts']

                self.vortstructs[i]['trcl'] = trcl
                self.vortstructs[i]['tinit'].set(vortstruct['eventbuff']['tinit'][:, np.newaxis, :])
                self.vortstructs[i]['state'].set(vortstruct['eventbuff']['state'][:, np.newaxis, :])
        
        self.tnext = min(etype['trcl'] for etype in self.vortstructs)

    def getvortbuff(self):
        pcg32rng = pcg32rxs_m_xs(self.seed)
        vid = 0
        temp = []
        tinits = []

        while vid < self.nvorts:
            tinits.append(self.tstart + 2*self.ls*pcg32rng.random()/self.avgu)
            vid += 1

        while any(tinit <= self.tend for tinit in tinits):     
            for vid, tinit in enumerate(tinits):
                state = pcg32rng.getstate()
                yinit = self.ymin + self.ydim*pcg32rng.random()
                zinit = self.zmin + self.zdim*pcg32rng.random()
                if tinit+(2*self.ls/self.avgu) >= self.tbegin and tinit <= self.tend:
                    temp.append((yinit, zinit, tinit, state))
                tinits[vid] += 2*self.ls/self.avgu
    
        return np.asarray(temp, self.vortbuffdtype)

    def getvortstructs(self, intg):
        vortstructs = []
        for etype, eles in intg.system.ele_map.items(): 
            neles = eles.neles
            pts = np.swapaxes(eles.ploc_at_np('upts'), 1, 0)
            ptsr = (self.rot @ (pts.reshape(3, -1) - self.shift[:,None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)
            inside = self.bbox.pts_in_region(ptsr)

            temp = defaultdict(list)
            stream = defaultdict()

            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0]
                ptsri = ptsr[:,eids,:] 
                insert = [(i, [*p.min(axis=0), *p.max(axis=0)], None)
                          for i, p in enumerate(ptsri.swapaxes(0, 1))]
                idx3d = Index(insert, properties=Property(dimension=3, interleaved=True))
                
                for vid, vort in enumerate(self.vortbuff):
                    elestemp = []
                    vboxmin = [-2*self.ls, vort['yinit']-self.ls, vort['zinit']-self.ls]
                    vboxmax = [2*self.ls, vort['yinit']+self.ls, vort['zinit']+self.ls]

                    candidate = np.array(list(set(idx3d.intersection((*vboxmin,*vboxmax,)))))   

                    vbox = BoxRegion(vboxmin, vboxmax)

                    if np.any(candidate):
                        vinside = vbox.pts_in_region(ptsri[:,candidate,:])
                        if np.any(vinside):
                            elestemp = np.any(vinside, axis=0).nonzero()[0].tolist()

                    for leid in elestemp:
                        exmin = ptsri[vinside[:,leid],candidate[leid],0].min()
                        exmax = ptsri[vinside[:,leid],candidate[leid],0].max()
                        ts = max(vort['tinit'], vort['tinit'] + (exmin/self.avgu))
                        te = max(ts,min(ts + (exmax-exmin+2*self.ls)/self.avgu,vort['tinit']+(2*self.ls/self.avgu)))
                        temp[eids[candidate[leid]]].append((vid,ts,te))

                for k, v in temp.items():
                    stream[k] = np.sort(np.asarray(v, self.streamdtype), order='ts')

                nvmx = 0
                for v in stream.values():
                    for i, ts in enumerate(v['ts']):
                        cnt = 0
                        while i-cnt >= 0:
                            if v['te'][i-cnt] < ts:
                                break
                            cnt += 1
                        nvmx = max(nvmx, cnt)

                nvmx += 1

                eventbuff = np.zeros((nvmx, neles), self.eventbuffdtype)

                vortstruct = {'trcl': 0.0, 'stream': stream, 'eventbuff': eventbuff,
                        'tinit': intg.backend.matrix((nvmx, 1, neles), tags={'align'}),
                        'state': intg.backend.matrix((nvmx, 1, neles), tags={'align'}, dtype=np.uint32)}

                eles.add_src_macro('pyfr.plugins.kernels.turbulence',
                                   'turbulence',
                                   self.macro_params | {'nvmx': nvmx},
                                   ploc=True, soln=True)

                eles._set_external('tinit',
                                   f'in broadcast-col fpdtype_t[{nvmx}][1]',
                                   value=vortstruct['tinit'])
                                   
                eles._set_external('state',
                                   f'in broadcast-col uint32_t[{nvmx}][1]',
                                   value=vortstruct['state'])

                vortstructs.append(vortstruct)

        return vortstructs
