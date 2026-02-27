from collections import defaultdict, namedtuple
import math

import numpy as np
from numpy.core.records import fromrecords
from rtree.index import Index, Property

from pyfr.plugins.base import BaseSolverPlugin
from pyfr.regions import BoxRegion

import time # Customized Check

VortStruct = namedtuple('vortstruct',
                        ['strms', 'strmsid', 'buf', 'tinit', 'state'])


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


class AnisotropicTurbulencePlugin(BaseSolverPlugin):
    name = 'anisotropicturbulence'
    systems = ['ac-navier-stokes', 'navier-stokes']
    formulations = ['dual', 'std']
    dimensions = [3]

    def __init__(self, intg, cfgsect):
        print("[Plugin] SEM starts", flush=True) # Customized Check
        super().__init__(intg, cfgsect)
        t0 = time.time()        # Customized Check

        self.fdptype = fdptype = intg.backend.fpdtype

        ac = intg.system.name.startswith('ac')

        self.nmvxinit = 6
        self.nvmxtol = 0.1
        self.tstart = intg.tstart
        self.tbegin = intg.tcurr
        self.tnext = intg.tcurr
        self.tend = intg.tend
        self.seed = 42

        self.eventdtype = [('tinit', fdptype), ('state', np.uint32),
                           ('ts', fdptype), ('te', fdptype)]

        # 1. Reference langth scale
        ls = self.cfg.getliteral(cfgsect, 'turbulence-length-scale')
        ls_array = np.array(ls, dtype=np.float64)

        if ls_array.ndim == 0 :
            self.mode = 'isotropic'
            self.l_ref = np.ones(3)*float(ls_array)
            self.l_tensor = np.ones((3,3))*float(ls_array)
        elif ls_array.ndim == 1 and ls_array.size == 3 :
            self.mode = 'partial_anisotropic'
            self.l_ref = ls_array
            self.l_tensor = (np.ones((3, 3))*ls_array).T
        elif ls_array.ndim == 2 and ls_array.shape == (3, 3) :
            self.mode = 'anisotropic'
            self.l_ref = np.max(ls_array, axis=1)
            self.l_tensor = ls_array
        else:
            raise ValueError("Invalid format for 'turbulence-length-scale'."
                             "Must be scalar, 3-vector, or 3x3 matrix.") 

        xi_min_max = self.cfg.getfloat(cfgsect, 'xi-min-max', 0.4) # xi_min_max (Giangaspero (2022) recommends 0.4)
        
        lref = np.asarray(self.l_ref, dtype=np.float64)
        if np.any(lref <= 0):
            raise ValueError("Reference length scales l_ref must be > 0.")
        
        Lij = np.asarray(self.l_tensor, dtype=np.float64) # Ensure l_tensor is (3,3) for unified handling
        if Lij.shape != (3, 3):
            # your code currently fills l_tensor even for isotropic/partial modes,
            # but keep a guard anyway
            Lij = np.ones((3, 3)) * float(np.mean(lref))

        # (Giangaspero (2022) xi_max[i,j] = max(l_ij / l_ref_i, xi_min_max) 
        xi_max = Lij / lref[:, None]
        self.xi_max = xi_max = np.maximum(xi_max, xi_min_max)
     
        # 2. Reynolds stress tensor
        ti_raw = self.cfg.get(cfgsect, 'turbulence-intensity', None)
        if ti_raw is not None and str(ti_raw).lower() != 'none':
            ti = float(ti_raw)
        else:
            ti = None
            
        rs = self.cfg.getliteral(cfgsect, 'reynolds-stress', None)      
        self.avgu = avgu = self.cfg.getfloat(cfgsect, 'avg-u')
        
        if rs is not None : # Giangaspero (2022)
            r_tensor = np.array(rs, dtype=np.float64)
            if r_tensor.shape != (3, 3) :
                raise ValueError("'reynolds-stress' must be a 3x3 matrix "
                                "((Rxx, Rxy, Rxz), (Ryx, Ryy, Ryz), (Rzx, Rzy, Rzz)).") 
            self.r_tensor = 0.5*(r_tensor + r_tensor.T)           
        elif ti is not None :
            self.r_tensor = np.eye(3)*(0.01*ti*self.avgu)**2
        else :
            raise ValueError("Either 'turbulence-intensity' or 'reynolds-stress' must be set.")
        
        # 3. Normalization & Cholesky decomposition
        sigma = self.cfg.getfloat(cfgsect, 'sigma')   
        gc = (2.0*sigma/(math.erf(1.0/sigma)*math.pi**0.5))**0.5
        norm_factor = (gc/sigma)**3
        
        try:
            L = np.linalg.cholesky(self.r_tensor)
        except np.linalg.LinAlgError:
            raise ValueError("Reynolds stress tensor must be positive definite.")
        
        st_correction = 1.0 / math.sqrt(2.0) # Correction factor for integration of source term
        
        beta3_tensor = L*norm_factor
        beta1_vector = -0.5/(sigma*self.l_ref)**2
        
        # 4. Other parameters depending on artificial compressiblity (AC) mode or not
        if not ac:
            gamma = self.cfg.getfloat('constants', 'gamma')
            avgrho = self.cfg.getfloat(cfgsect, 'avg-rho')
            avgmach = self.cfg.getfloat(cfgsect, 'avg-mach')
            beta2 = avgrho * (gamma - 1) * avgmach**2 * st_correction
        else:
            beta2 = 0

        ydim = self.cfg.getfloat(cfgsect, 'y-dim')
        zdim = self.cfg.getfloat(cfgsect, 'z-dim')
        self.yzdim = yzdim = np.array([ydim, zdim])
        
        self.nvorts = int(ydim*zdim / (4*self.l_ref[1]*self.l_ref[2]))
        self.bbox = BoxRegion([-2*self.l_ref[0], *(-0.5*yzdim - self.l_ref[1:])],
                              [ 2*self.l_ref[0], *( 0.5*yzdim + self.l_ref[1:])])

        centre = self.cfg.getliteral(cfgsect, 'centre')
        rax = np.array(self.cfg.getliteral(cfgsect, 'rot-axis'))
        ran = np.radians(self.cfg.getfloat(cfgsect, 'rot-angle'))

        self.shift = shift = np.array(centre)
        self.rot = rot = (np.cos(ran)*np.eye(3) +
                          np.sin(ran)*(np.cross(rax, np.eye(3))) +
                          (1 - np.cos(ran))*np.outer(rax, rax))

        self.macro_params = {
            'ls': self.l_ref.tolist(), 
            'avgu': avgu, 
            'yzdim': yzdim, 
            'beta1': beta1_vector.tolist(),
            'beta2': beta2, 
            'beta3': beta3_tensor.tolist(),
            'xi_max': self.xi_max.tolist(), 
            'rot': rot, 
            'shift': shift,
            'ac': ac
        }

        self.trcl = {}
        
        print("[Plugin] SEM inserting buffer time starts", flush=True)
        t1 = time.time()       # Customized Check
        self.vortbuf = self.get_vort_buf()
        t2 = time.time()       # Customized Check
        print(f"[Plugin] SEM inserting buffer time done : {t2-t1}s", flush=True)
        
        print("[Plugin] SEM generating eddy structures start", flush=True)
        t3 = time.time()
        self.vortstructs = self.get_vort_structs(intg)
        t4 = time.time()        # Customized Check
        print(f"[Plugin] SEM generating eddy structures done: {t4-t3}s", flush=True)
        print(f"[Plugin] SEM __init__ FINISH, total={t4 - t0:.2f} s", flush=True) # Customized Check

        if not bool(self.vortstructs):
            self.tnext = float('inf')

    def __call__(self, intg):
        if intg.tcurr + intg._dt < self.tnext:
            return

        for etype, vs in self.vortstructs.items():
            if self.trcl[etype] > self.tnext:
                continue

            self.trcl[etype] = self.update_buf(vs.strms, vs.strmsid, vs.buf,
                                               self.tnext)
            vs.tinit.set(vs.buf.tinit)
            vs.state.set(vs.buf.state)

        self.tnext = min(self.trcl.values())

    def update_buf(self, strms, strmsid, buf, tnext):
        trcltmp = float('inf')
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

            if sft < strm.shape[0] and buf[-1, ele].ts < trcltmp:
                trcltmp = buf[-1, ele].ts

            strmsid[ele] = sft

        return trcltmp

    def test_nvmx(self, strms, neles, nvmx):
        tnext = self.tnext
        strmsid = {ele: 0 for ele in strms}
        buf = fromrecords(np.zeros((nvmx, neles)), dtype=self.eventdtype)

        while tnext < float('inf'):
            trcltmp = self.update_buf(strms, strmsid, buf, tnext)
            if trcltmp < tnext + self.nvmxtol:
                return False

            tnext = trcltmp

        return True

    def get_vort_buf(self):
        dt  = 2*self.l_ref[0]/self.avgu
        pcg = pcg32rxs_m_xs(self.seed)

        tmp = []
        tinits = [self.tstart + dt*next(pcg)[1] for vid in range(self.nvorts)]

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

    def get_vort_structs(self, intg):
        lx, ly, lz = self.l_ref
        avgu = self.avgu
        
        props = Property(dimension=3, interleaved=True)
        eventdtype = self.eventdtype
        vortstructs = {}

        for etype, eles in intg.system.ele_map.items():
            neles = eles.neles
            pts = eles.ploc_at_np('upts').swapaxes(0, 1)
            ptsr = (self.rot @ (pts.reshape(3, -1) -
                    self.shift[:, None])).reshape(pts.shape)
            ptsr = np.moveaxis(ptsr, 0, -1)

            inside = self.bbox.pts_in_region(ptsr)
            if not np.any(inside):
                continue

            tmp = defaultdict(list)
            strms = {}
            strmsid = {}

            eids = np.any(inside, axis=0).nonzero()[0]
            ptsri = ptsr[:, eids]
            insert = [(i, [*p.min(axis=0), *p.max(axis=0)], None)
                      for i, p in enumerate(ptsri.swapaxes(0, 1))]
            rtree = Index(insert, properties=props)

            for vid, vort in enumerate(self.vortbuf):
                vbmin = [-2*lx, vort.yinit - ly, vort.zinit - lz]
                vbmax = [ 2*lx, vort.yinit + ly, vort.zinit + lz]

                # Get candidate points from the R-tree
                isect = rtree.intersection((*vbmin, *vbmax))
                cands = np.array(list(set(isect)), dtype=int)

                # Get exact points candidate points
                boxrgn = BoxRegion(vbmin, vbmax)
                vinside = boxrgn.pts_in_region(ptsri[:, cands])

                for vi, c in zip(vinside.swapaxes(0, 1), cands):
                    if not np.any(vi):
                        continue

                    xv = ptsri[vi, c, 0]
                    xvmin, xvmax = xv.min(), xv.max()

                    ts = max(vort.tinit, vort.tinit + xvmin/avgu)
                    
                    te = max(ts, min(ts + (xvmax - xvmin + 2*lx)/avgu,
                                     vort.tinit + 2*lx/avgu))

                    # Record when a vortex enters and exits an element
                    tmp[eids[c]].append((self.vortbuf[vid].tinit,
                                         self.vortbuf[vid].state, ts, te))

            for ele, strm in tmp.items():
                strms[ele] = np.sort(fromrecords(strm, dtype=eventdtype),
                                     order='ts')
                strmsid[ele] = 0

            nvmx = self.nmvxinit
            while not self.test_nvmx(strms, neles, nvmx):
                nvmx += 1

            eles.add_src_macro(
                'pyfr.plugins.kernels.anisotropicturbulence', 'anisotropicturbulence',
                self.macro_params | {'nvmx': nvmx}, ploc=True, soln=True
            )

            buf = fromrecords(np.zeros((nvmx, neles)), dtype=eventdtype)

            self.trcl[etype] = 0
            vortstructs[etype] = vs = VortStruct(
                strms, strmsid, buf,
                intg.backend.matrix((nvmx, neles), tags={'align'}),
                intg.backend.matrix((nvmx, neles), tags={'align'},
                                    dtype=np.uint32)
            )

            eles._set_external('tinit', f'in broadcast-col fpdtype_t[{nvmx}]',
                               value=vs.tinit)
            eles._set_external('state', f'in broadcast-col uint32_t[{nvmx}]',
                               value=vs.state)

        return vortstructs
