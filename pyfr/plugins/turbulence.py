import math
import random
import time
import uuid

import numpy as np

from collections import defaultdict
from pyfr.plugins.base import BaseSolverPlugin
from pyfr.regions import BoxRegion
from rtree.index import Index, Property


class pcg32rxs_m_xs:
    def __init__(self, seed):
        self.state = np.uint32(seed)
        self.multiplier = np.uint32(747796405)
        self.mcgmultiplier = np.uint32(277803737)
        self.increment = np.uint32(2891336453)
        self.opbits = np.uint8(4)
        self.b8 = np.uint8(8)
        self.b22 = np.uint8(22)
        self.b28 = np.uint8(28)
        self.b32 = np.uint8(32)
    def rand(self):
        oldstate = self.state
        self.state = (oldstate * self.multiplier) + self.increment
        rshift = np.uint8(oldstate >> self.b28)
        oldstate ^= oldstate >> (self.opbits + rshift)
        oldstate *= self.mcgmultiplier
        oldstate ^= oldstate >> self.b22
        return oldstate
    def random(self):
        return (self.rand() >> self.b8) * 5.9604644775390625e-8
    def getstate(self):
        return self.state


class TurbulencePlugin(BaseSolverPlugin):
    name = 'turbulence'
    systems = ['navier-stokes']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend

        self.seed = self.cfg.getint(cfgsect, 'seed')

        fdptype = intg.backend.fpdtype

        self.vortdtype = np.dtype([('loci', fdptype, 2), ('tinit', fdptype), ('state', np.uint32)])
        self.sstreamdtype = np.dtype([('vid', '<i4'), ('ts', fdptype), ('te', fdptype)])
        self.buffdtype = np.dtype([('tinit', fdptype), ('state', np.uint32), ('ts', fdptype), ('te', fdptype)])

        gamma = self.cfg.getfloat('constants', 'gamma')
        rhobar = self.cfg.getfloat(cfgsect, 'rho-bar')
        self.ubar = self.cfg.getfloat(cfgsect, 'u-bar')
        machbar = self.cfg.getfloat(cfgsect, 'mach-bar')
        rootrs = self.cfg.getfloat(cfgsect, 'turbulence-intensity')*self.ubar/100.0
        sigma = self.cfg.getfloat(cfgsect, 'sigma')
        self.ls = self.cfg.getfloat(cfgsect, 'turbulence-length-scale')

        srafac = rhobar*(gamma-1.0)*machbar*machbar
        gc = math.sqrt((2.0*sigma/(math.sqrt(math.pi)))*(1.0/math.erf(1.0/sigma)))

        fac = (rootrs*gc*gc*gc)/(sigma*sigma*sigma)

        ydim = self.cfg.getfloat(cfgsect,'y-dim')
        zdim = self.cfg.getfloat(cfgsect,'z-dim')
        
        self.xmin = - self.ls
        self.xmax = self.ls
        self.ymin = -ydim/2
        self.ymax = ydim/2
        self.zmin = -zdim/2
        self.zmax = zdim/2

        self.nvorts = int((self.ymax-self.ymin)*(self.zmax-self.zmin)/(4*self.ls*self.ls))
        self.bbox = BoxRegion([self.xmin-self.ls,self.ymin-self.ls,self.zmin-self.ls],
                         [self.xmax+self.ls,self.ymax+self.ls,self.zmax+self.ls])
        
        theta = -1.0*np.radians(self.cfg.getfloat(cfgsect,'rot-angle'))

        c = self.cfg.getliteral(cfgsect, 'centre')
        e = np.array(self.cfg.getliteral(cfgsect, 'rot-axis'))

        self.shift = np.array(c)
        self.rot=(np.cos(theta)*np.identity(3))+(np.sin(theta)*(np.cross(e, np.identity(3) * -1)))+(1.0-np.cos(theta))*np.outer(e,e)

        self.dtol = 0
        
        if hasattr(intg, 'dtmax'):
            self.dtol = intg.dtmax
        else:
            self.dtol = intg._dt

        self.macro_params = {'ls': self.ls, 'ubar': self.ubar, 'srafac': srafac,
                 'ymin': self.ymin, 'ymax': self.ymax, 'zmin': self.zmin, 'zmax': self.zmax,
                 'sigma' : sigma, 'rootrs': rootrs, 'gc': gc, 'fac': fac, 'rot': self.rot, 'shift': self.shift
                }

        self.vortbuff = self.vortbuff()
        self.actbuffs = self.actbuffs(intg)

        if not bool(self.actbuffs):
           self.tnext = float('inf')
                  
    def __call__(self, intg):
        tcurr = intg.tcurr
        if tcurr+self.dtol >= self.tnext:
            for abid, actbuff in enumerate(self.actbuffs):    
                if actbuff['trcl'] <= self.tnext:
                    trcl = np.inf
                    for geid, sstream in actbuff['sstream'].items():
                        if sstream['vid'].any():
                            tmp = actbuff['buff'][:,geid][actbuff['buff'][:,geid]['te'] > tcurr]        
                            shft = actbuff['nvmx']-len(tmp)   
                            if shft:
                                newb = np.zeros(shft, self.buffdtype)
                                temp = self.vortbuff[['tinit', 'state']][sstream['vid'][:shft]]
                                pad = shft-temp.shape[0]
                                newb[['tinit', 'state']] = np.pad(temp, (0,pad), 'constant')
                                newb[['ts', 'te']] = np.pad(sstream[['ts', 'te']][:shft], (0,pad), 'constant')
                                self.actbuffs[abid]['buff'][:,geid] = np.concatenate((tmp,newb))
                                self.actbuffs[abid]['sstream'][geid] = sstream[shft:]
                            else:
                                tstemp = sstream['ts'][0]
                                if tcurr >= tstemp:
                                    print(f'Active vortex in stream cannot fit on buffer.')
                                
                            if self.actbuffs[abid]['sstream'][geid]['vid'].any() and (self.actbuffs[abid]['buff'][-1,geid]['ts'] < trcl):
                                trcl = self.actbuffs[abid]['buff'][-1,geid]['ts']

                    self.actbuffs[abid]['trcl'] = trcl
                    self.actbuffs[abid]['tinit'].set(actbuff['buff']['tinit'][:, np.newaxis, :])
                    self.actbuffs[abid]['state'].set(actbuff['buff']['state'][:, np.newaxis, :])
            
            proptnext = min(etype['trcl'] for etype in self.actbuffs)
            if proptnext > self.tnext:
                self.tnext = proptnext
            else:
                print('Not advancing.')

    def vortbuff(self):
        pcg32rng = pcg32rxs_m_xs(self.seed)
        vid = 0
        temp = []
        xtemp = []
        
        tinits = []
        
        while vid < self.nvorts:
            tinits.append(self.tstart + (self.xmax-self.xmin)*pcg32rng.random()/self.ubar)
            vid += 1

        while True:     
            for vid, tinit in enumerate(tinits):
                state = pcg32rng.getstate()
                yinit = self.ymin + (self.ymax-self.ymin)*pcg32rng.random()
                zinit = self.zmin + (self.zmax-self.zmin)*pcg32rng.random()
                if tinit+((self.xmax-self.xmin)/self.ubar) >= self.tbegin and tinit <= self.tend:
                    xtemp.append(((yinit,zinit),tinit,state))
                tinits[vid] += (self.xmax-self.xmin)/self.ubar
            if all(tinit > self.tend for tinit in tinits):
                break
    
        return np.asarray(xtemp, self.vortdtype)

    def actbuffs(self, intg):
        actbuffs = []
        for etype, eles in intg.system.ele_map.items(): 
            neles = eles.neles
            pts = eles.ploc_at_np('upts')
            pts = np.moveaxis(pts, 1, 0)
            ptsr = (self.rot @ (pts.reshape(3, -1) - self.shift[:,None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)
            inside = self.bbox.pts_in_region(ptsr)

            stream = defaultdict(list)
            sstream = defaultdict()

            if np.any(inside):
                eids = np.any(inside, axis=0).nonzero()[0] # eles in injection box
                ptsri = ptsr[:,eids,:] # points in injection box

                props = Property(dimension=3, interleaved=True)
                idx3d = Index(properties=props)
                
                nleles = ptsri.shape[1]

                for i in range(nleles):
                    idx3d.insert(i,(ptsri[:,i,0].min(),ptsri[:,i,1].min(),ptsri[:,i,2].min(),
                                    ptsri[:,i,0].max(),ptsri[:,i,1].max(),ptsri[:,i,2].max()))

                for vid, vort in enumerate(self.vortbuff):
                    vbox = BoxRegion([self.xmin-self.ls, vort['loci'][0]-self.ls, vort['loci'][1]-self.ls],
                                     [self.xmax+self.ls, vort['loci'][0]+self.ls, vort['loci'][1]+self.ls])

                    elestemp = []
                    candidate = np.array(list(set(idx3d.intersection((self.xmin-self.ls, vort['loci'][0]-self.ls, vort['loci'][1]-self.ls,
                                                                      self.xmax+self.ls, vort['loci'][0]+self.ls, vort['loci'][1]+self.ls)))))   

                    if np.any(candidate):
                        vinside = vbox.pts_in_region(ptsri[:,candidate,:])
                        if np.any(vinside):
                            elestemp = np.any(vinside, axis=0).nonzero()[0].tolist()

                    for leid in elestemp:
                        exmin = ptsri[vinside[:,leid],candidate[leid],0].min()
                        exmax = ptsri[vinside[:,leid],candidate[leid],0].max()
                        ts = max(vort['tinit'], vort['tinit'] + ((exmin - self.xmin - self.ls)/self.ubar))
                        te = max(ts,min(ts + (exmax-exmin+2*self.ls)/self.ubar,vort['tinit']+((self.xmax-self.xmin)/self.ubar)))
                        stream[eids[candidate[leid]]].append((vid,ts,te))

                for k, v in stream.items():
                    v.sort(key=lambda x: x[1]) 
                    sstream[k] = np.asarray(v, self.sstreamdtype)

                nvmx = 0
                for leid, actl in sstream.items():
                    for i, ts in enumerate(actl['ts']):
                        cnt = 0
                        while i-cnt >= 0:
                            if actl['te'][i-cnt] < ts:
                                break
                            cnt += 1
                        if cnt > nvmx:
                            nvmx = cnt
                nvmx += 1

                buff = np.zeros((nvmx, neles), self.buffdtype)

                actbuff = {'trcl': 0.0, 'sstream': sstream, 'nvmx': nvmx, 'buff': buff,
                           'tinit': eles._be.matrix((nvmx, 1, neles), tags={'align'}),
                           'state': eles._be.matrix((nvmx, 1, neles), tags={'align'}, dtype=np.uint32)}

                eles.add_src_macro('pyfr.plugins.kernels.turbulence','turbulence', self.macro_params | {'nvmax': nvmx}, ploc=True, soln=True)

                eles._set_external('tinit',
                                   f'in broadcast-col fpdtype_t[{nvmx}][1]',
                                   value=actbuff['tinit'])
                                   
                eles._set_external('state',
                                   f'in broadcast-col uint32_t[{nvmx}][1]',
                                   value=actbuff['state'])

                actbuffs.append(actbuff)

        return actbuffs

