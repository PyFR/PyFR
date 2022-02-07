# -*- coding: utf-8 -*-

import os, sys
import re
import numpy as np
from collections import defaultdict
from mpi4py import MPI
from functools import cached_property

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


def open_csv(fname, header=None, mode='w'):
    # Open for mode
    outf = open(fname, mode)
    # Output a header if required
    if os.path.getsize(fname) == 0 and header:
        print(header, file=outf)
    return outf

def write_fftdata(fname, *indata, header='', mode='a'):
    #open the file
    outf = open_csv(fname, ''.join(header), mode)
    #prepare the data array
    data = np.array(indata)
    dshape = data.shape
    #do write
    for i in range(dshape[1]):
        print(*data[:,i].ravel(), sep=' ', file=outf)
    outf.flush()
    return

#-FFT utilities
def rfft(udata_):
    dsize = np.size(udata_)
    # freqsize: size of half the spectra with positive frequencies
    freqsize = int(dsize/2) if dsize%2 == 0 else int(dsize-1)/2
    ufft = np.fft.rfft(udata_)/freqsize 
    return ufft

def compute_spl(pref,amp,df=None):
    sqrt2 = 1./np.sqrt(2.) 
    lgsqrt2 = 20.*np.log10(sqrt2)
    factor = sqrt2/pref if not df else sqrt2/(pref*np.sqrt(df))
    spl = 20.*np.log10(factor*amp) 
    spl[0] -= lgsqrt2
    spl[-1] -= lgsqrt2
    #overall sound pressure level
    ospl_sum = sum(pow(10.,spl/10.)) if not df else df*sum(pow(10.,spl/10.))
    oaspl = 10.* np.log10(ospl_sum) 
    return spl, oaspl

def compute_psd(amp,df):
    psd = 0.5 * amp * amp / df
    psd[0] *= 2.
    psd[-1] *= 2.
    return psd

def welch_accum(pfft,pmagsum,presum,pimgsum,scale_mode='density'):
    mag = np.abs(pfft)*np.abs(pfft) if scale_mode == 'density' else np.abs(pfft)
    pmagsum += mag
    presum  += np.real(pfft)
    pimgsum += np.imag(pfft)
    return pmagsum,presum,pimgsum

def welch_average(aver_count,pmagsum,presum,pimgsum,scale_mode='density'):
    mag = np.sqrt(pmagsum/aver_count) if scale_mode=='density' else pmagsum/aver_count
    phase = np.arctan2(pimgsum,presum)
    pfft = mag * np.exp(1j * phase)
    return pfft

def get_num_nearest_pow_of_two(N):
    exponent = int(np.log2(N))
    diff0 = np.abs(N-pow(2,exponent))
    diff1 = np.abs(N-pow(2,exponent+1))
    return pow(2,exponent) if diff0 < diff1 else pow(2,exponent+1)


class FwhSolverPlugin(PostactionMixin, BasePlugin):

    name = 'fwhsolver'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']

    # refernce pressure for spl computation
    pref = 2.e-5

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.DEBUG = self.cfg.getint(cfgsect,'debug',0)
        # Base output directory and file name
        self.basedir = basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        self.basename = self.cfg.get(self.cfgsect, 'basename')
        
        # read time and fft parameters
        self.tstart = self.cfg.getfloat(cfgsect,'tstart',intg.tstart)
        dt_sub = self.cfg.getfloat(cfgsect,'dt',intg._dt)
        # read the window length, default is up to simulation tend
        ltsub_def = intg.tend - self.tstart
        ltsub = self.cfg.getfloat(cfgsect,'Lt',ltsub_def)        
        shift = self.cfg.getfloat(cfgsect,'time-shift',0.5)
        window_func = self.cfg.get(cfgsect,'window','hanning')
        psd_scale_mode = self.cfg.get(cfgsect,'psd-scaling-mode','density')

        fftcls = FFTParam(intg._dt,dt_sub,ltsub,shift,window_func,psd_scale_mode)
        self._started = False
        self.steps_count = 0
        self.tout_last = self.tstart #intg.tcurr
        self.dt_out = fftcls.dt_sub
        self.samplesteps = fftcls.samplesteps
        # read observers inputs
        self.observ_locs = np.array(self.cfg.getliteral(self.cfgsect, 'observer-locs'))
        self.nobserv = self.observ_locs.shape[0]
        
        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]
        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Check if the system is incompressible
        self._artificial_compress = intg.system.name.startswith('ac')

        # read flow data
        self.constvars = self.cfg.items_as('constants', float)
        pvmap = intg.system.elementscls.privarmap[self.ndims]
        self.uinf = defaultdict(list)
        for pvvar in pvmap:
            if pvvar in ['u','v','w']:
                self.uinf['u'].append(npeval(self.cfg.getexpr(cfgsect, pvvar), self.constvars))
            else:
                self.uinf[pvvar] = npeval(self.cfg.getexpr(cfgsect, pvvar), self.constvars)
        if self._artificial_compress:
            self.uinf['rho'] = self.uinf['p']/self.constvars['ac-zeta']
        self.uinf['c'] = np.sqrt(self.uinf['p']/self.uinf['rho'])
        self.uinf['Mach'] = self.uinf['u']/self.uinf['c']

        # Region of interest
        box = self.cfg.getliteral(self.cfgsect, 'region')
        # Prepare fwh surface mesh data
        region_eset = self.extract_drum_region_eset(intg, box)
        self.prepare_surfmesh(intg,region_eset)

        # Determine fwh active ranks lists
        self.fwh_comm, self.fwh_edgecomm = self.build_fwh_subranks_lists(intg)        

        # Construct the fwh mesh writer
        self._writer = NativeWriter(intg, basedir, self.basename, 'soln')

        # write the fwh surface geometry file
        self._write_fwh_surface_geo(intg,self._inside_eset,self._eidxs)

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # Underlying elements class
        self.elementscls = intg.system.elementscls
        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # Extract FWH Surf Geometric Data:
        self._qpts_info, self._m0 = self.get_fwhsurf_finfo(elemap, self._eidxs)  #fpts, qpts_dA, norm_pnorms
        self._int_qpts, self._int_m0 = self.get_nqpts_m0(elemap, self._int_eidxs)
        self._mpi_qpts, self._mpi_m0 = self.get_nqpts_m0(elemap, self._mpi_eidxs)
        self._bnd_qpts, self._bnd_m0 = self.get_nqpts_m0(elemap, self._int_eidxs)
        self.nqpts = np.asarray(self._qpts_info[0]).shape[0]
        self.fwhnvars = len(self.uinf['u'])+2

        # Allocate/Prepare solution data arrays:
        # nvars = 5 if self.ndims == 3 else 4
        # self._fwh_usoln = self.allocate_usoln_darray(self.nqpts, fftcls.ntsub ,nvars)

        #Debug, using analytical sources
        test = self.cfg.get(self.cfgsect, 'analytic-src', None)
        self.pacoustsrc = None
        xyzshift = 0
        if self.active_fwhrank:
            if test == 'monopole':
                #shift the x,y,z coordinates so that the source is inside the body
                xyz_min = np.array([np.min(cr) for cr in self._qpts_info[0].T])
                xyz_max = np.array([np.max(cr) for cr in self._qpts_info[0].T])
                self.fwh_comm.Allreduce(get_mpi('in_place'),[xyz_min, MPI.DOUBLE],op=get_mpi('min'))
                self.fwh_comm.Allreduce(get_mpi('in_place'),[xyz_max, MPI.DOUBLE],op=get_mpi('max'))
                xyzshift = 0.5 *(xyz_min+xyz_max)
                #init the point src parameters
                srclocs = self.cfg.getliteral(self.cfgsect, 'src-locs',[0.,0.,0.])
                srclocs += xyzshift
                srcuinf = np.array([self.uinf['rho'],self.uinf['u'][0],self.uinf['u'][1],self.uinf['u'][2],self.uinf['p']])
                srcamp = self.cfg.getfloat(self.cfgsect,'src-amp',1.)
                srcfreq = self.cfg.getfloat(self.cfgsect,'src-freq',5.)
                tperiod = 1./srcfreq
                nperiods = fftcls.ntsub * fftcls.dt_sub / tperiod
                gamma = self.constvars['gamma']
                self.pacoustsrc = MonopoleSrc('smonopole',srclocs,srcuinf,srcamp,srcfreq,fftcls.ntsub,nperiods,gamma)

            # init the fwhsolver
            if np.any(xyzshift):
                self.observ_locs += xyzshift
            if self.active_fwhrank:
                self.fwhsolver = FwhSolver(self.observ_locs,self._qpts_info,self.uinf,fftcls)

            self.writemode = 'w' if not intg.isrestart else 'a'

    def __call__(self, intg):

        # if not an active rank return
        if not self.active_fwhrank:
            return
        # If we are not supposed to start fwh yet then return
        if intg.tcurr < self.tstart:
            return
        # Return if no usoln sampling is due
        dosample = intg.nacptsteps % self.samplesteps == 0
        if not dosample:
            return
        self._started = intg.tcurr >= (self.tstart - self.tol)

        # MPI info
        comm, rank, root = get_comm_rank_root()
        
        # sample solution if it is due
        if dosample:
            ushape = (self.fwhnvars,self.nqpts)
            windowtime = intg.tcurr - self.tstart - self.fwhsolver.naver * self.fwhsolver.fftcls.shift * self.fwhsolver.fftcls.ltsub
            srctime = intg.tcurr - self.tstart
            if self.pacoustsrc:
                self.fwhsolver.usoln_onestep = self.pacoustsrc.update_usoln_onestep(self._qpts_info[0], srctime, ushape) 
            else:
                self.fwhsolver.usoln_onestep = self.update_usoln_from_currstep(intg, self._m0, self._eidxs, ushape)
            self.steps_count = self.fwhsolver.nsteps
            
            #Debug
            if self.fwhrank == 0:
                print(f'FWH sampled, wstep {self.fwhsolver.nsteps}, wtime {np.round(windowtime,5)}, ctime {np.round(srctime,5)}',flush=True)
        
        # compute fwh solution if due
        docompute = self.fwhsolver.nsteps == self.fwhsolver.ntsub 
        if docompute:
            self.fwhsolver.compute_fwh_solution()
            self.steps_count = self.fwhsolver.nsteps
            self.nwindows = self.fwhsolver.naver-1 
            pfft = self.fwhsolver.pfft
            if self.fwhrank != 0:
                self.fwh_comm.Reduce(pfft, None, op=get_mpi('sum'), root=0)
            else:
                self.fwh_comm.Reduce(get_mpi('in_place'), pfft, op=get_mpi('sum'), root=0)
            
            # Debug
            if self.fwhrank == 0:
                print(f'FWH computed ........, Naver {self.nwindows}')

            #compute spectrums
            if self.fwhrank == 0:
                amp = np.abs(pfft)
                df = self.fwhsolver.freq[1]
                # computing power spectrum outputs
                psd = compute_psd(amp,df)
                spl, oaspl = compute_spl(self.pref,amp)
                # Writing fft outputs
                bname = os.path.join(self.basedir,f'{self.basename}'+'_ob{ob}.csv')
                for ob in range(self.nobserv):
                    naverarr = np.tile(str(self.nwindows),len(pfft[ob]))
                    wdata = np.array([self.fwhsolver.freq,np.abs(pfft[ob,:]),np.angle(pfft[ob,:]),psd[ob,:],spl[ob,:],naverarr]).T
                    fname = bname.format(ob=ob)
                    header = ','.join(['#Frequency (Hz)', ' Magnitude (Pa)', ' Phase (rad)'])
                    header += ', PSD (Pa^2/Hz)' if self.fwhsolver.fftcls.psd_scale_mode == 'density' else [' POWER-SPECTRUM (Pa^2/Hz)']
                    header += f', SPL (dB), Nwindow'
                    write_fftdata(fname,wdata,header=header,mode=self.writemode)

                #Debug, exact source solution
                if self.pacoustsrc:
                    freq_ex, pfft_ex = self.pacoustsrc.exact_solution(self.observ_locs)
                    bname = os.path.join(self.basedir,f'{self.basename}'+'_exact-ob{ob}.csv')
                    for ob in range(self.nobserv):
                        fname = bname.format(ob=ob)
                        header = ','.join(['#Frequency (Hz)', ' Magnitude (Pa)', ' Phase (rad)'])
                        write_fftdata(fname,freq_ex,np.abs(pfft_ex[ob,:]),np.angle(pfft_ex[ob,:]),header=header,mode=self.writemode)
                print(f'FWH written ............. ')
                self.writemode ='a'

            # Update the last output time
            self.tout_last = intg.tcurr

        # Need to print fwh surface geometric data only once
        #intg.completed_step_handlers.remove(self)
        #exit(0)

    def extract_drum_region_eset(self, intg, drum):
        elespts = {}
        eset = {}
        
        drum = np.asarray(drum)
        nstations = drum.shape[0]-1

        mesh = intg.system.mesh
        for etype in intg.system.ele_types:
            elespts[etype] = mesh[f'spt_{etype}_p{intg.rallocs.prank}']

        slope = []
        for i in range(0,nstations):
            slope.append((drum[i+1,1]-drum[i,1])/(drum[i+1,0]-drum[i,0]))
        
        # shift the z origin to have a symmetric drum around the origin
        pz_max = -1e20
        pz_min = 1e20
        pz_shift=0.
        if self.ndims==3:
            for etype in intg.system.ele_types:
                pts = np.moveaxis(elespts[etype],2,0)
                for px,py,pz in zip(pts[0],pts[1],pts[2]):
                    pz_max = np.max([pz_max, np.max(pz)])
                    pz_min = np.min([pz_min, np.min(pz)])

            #if pz_min > 1.e-16: 
            pz_shift = (pz_min+pz_max)*.5 

        for etype in intg.system.ele_types:
            pts = np.moveaxis(elespts[etype],2,0)
            
            # Determine which points are inside the fwh surface
            inside = np.zeros(pts.shape[1:], dtype=np.bool)

            if self.ndims==3:
                for px,py,pz in zip(pts[0],pts[1],pts[2]):
                    pz -= pz_shift
                    for i in range(0,nstations):
                        inside += ((drum[i,0] <= px) & (px <= drum[i+1,0])) & (np.sqrt(py*py+pz*pz) < (drum[i,1] + (px-drum[i,0])*slope[i]))
            else:
                for px,py in zip(pts[0],pts[1]):
                    for i in range(0,nstations):
                        inside +=  ((drum[i,0] <= px) & (px <= drum[i+1,0])) & (np.abs(py) < (drum[i,1] + (px-drum[i,0])*slope[i]))

            if np.sum(inside):
                eset[etype] = np.any(inside, axis=0).nonzero()[0]

        return eset

    def _write_fwh_surface_geo(self,intg,eset,inters_eidxs):
        comm, rank, root = get_comm_rank_root()

        # If we are the root rank then prepare the metadata
        if rank == root:
            stats = Inifile()
            stats.set('data', 'fields', ','.join(self.fields))
            stats.set('data', 'prefix', 'soln')
            intg.collect_stats(stats)
            metadata = dict(intg.cfgmeta,
                            stats=stats.tostr(),
                            mesh_uuid=intg.mesh_uuid)
        else:
            metadata = None

        # Fetch and (if necessary) subset the solution
        ele_regions, ele_region_data = [], {}
        for etype, eidxs in sorted(eset.items()):
            doff = intg.system.ele_types.index(etype)
            darr = np.unique(eidxs).astype(np.int32)

            ele_regions.append((doff, etype, darr))
            ele_region_data[f'{etype}_idxs'] = darr

        data = dict(ele_region_data)
        for idx, etype, rgn in ele_regions:
            data[etype] = intg.soln[idx][..., rgn].astype(self.fpdtype)

        # Write out the file
        self.fwhmeshfname = self._writer.write(data, intg.tcurr, metadata)

        # Construct the VTU writer
        vtuwriter = VTUSurfWriter(intg,inters_eidxs)
        # Write VTU file:
        vtumfname = os.path.splitext(self.fwhmeshfname)[0]
        vtuwriter.write(vtumfname)

    def prepare_surfmesh(self,intg,eset):
        self._inside_eset = defaultdict(list)
        self._eidxs= defaultdict(list)  
        self._int_eidxs = defaultdict(list)
        self._mpi_eidxs = defaultdict(list)  
        self._mpi_sbuf_eidxs = defaultdict(list)
        self._bnd_eidxs = defaultdict(list) 
        self._intinters = []
        self._mpiinters = []
        self._bndinters = []
        self._inters = []
        self._ninters = {}
        self._nintinters = {}
        self._nmpiinters = {}
        self._nbndinters = {}
        self.ftype_index = {}
        # fwhranks own all fwh edges and hence are the ones that perform acoustic solve
        # fwhedgeranks are ranks who touches an fwh edge but not necessarily
        # owns an edge in general.
        # However, they can be in both fwhedgeranks & fwhranks list if they happen to be both touching 
        # an edge as an outsider and has inside cells (insiders) and hence own some edges
        self.fwhranks_list = []     
        self.fwh_edgranks_list = []

        mesh = intg.system.mesh
        # Collect fwh interfaces and their info
        self.collect_intinters(intg.rallocs,mesh,eset,intg.system.ele_map)
        self.collect_mpiinters(intg.rallocs,mesh,eset)
        self.collect_bndinters(intg.rallocs,mesh,eset)

        for index, (etype,fidx) in enumerate(self._eidxs):
            self.ftype_index[etype,fidx] = index
            if (etype,fidx) in self._nintinters:
                self._ninters[etype,fidx] = self._nintinters[etype,fidx] 
            if (etype,fidx) in self._nmpiinters:
                if (etype,fidx) in self._ninters:
                    self._ninters[etype,fidx] += self._nmpiinters[etype,fidx]
                else:
                    self._ninters[etype,fidx] = self._nmpiinters[etype,fidx]
            if (etype,fidx) in self._nbndinters:
                if (etype,fidx) in self._ninters:
                    self._ninters[etype,fidx] += self._nbndinters[etype,fidx] 
                else:
                    self._ninters[etype,fidx] = self._nbndinters[etype,fidx]

        eidxs = self._eidxs
        self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
        self._inters = np.array(self._inters)
        self.Numallinters = self._inters.shape[0]

    def collect_intinters(self,rallocs,mesh,eset,elemap):
        prank = rallocs.prank
        flhs, frhs = mesh[f'con_p{prank}'].astype('U4,i4,i1,i2').tolist()
        for ifaceL,ifaceR in zip(flhs,frhs):    
            etype, eidx = ifaceR[0:2]
            flagR = (eidx in eset[etype]) if etype in eset else False
            etype, eidx = ifaceL[0:2]
            flagL = (eidx in eset[etype]) if etype in eset else False
            # interior faces
            if (not (flagL & flagR)) & (flagL | flagR):
                if flagL:
                    fidx = ifaceL[2]
                    self._intinters.append([ifaceL[0:3],ifaceR[0:3]])
                    self._int_eidxs[etype,fidx].append(eidx)

                else:
                    etype, eidx, fidx = ifaceR[0:3]
                    self._intinters.append([ifaceR[0:3],ifaceL[0:3]])
                    self._int_eidxs[etype,fidx].append(eidx)
                self._eidxs[etype,fidx].append(eidx)
                self._inters.append([etype,fidx,eidx])
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)
            # periodic faces:
            elif (flagL & flagR) & (ifaceL[3] != 0 ):   
                etype, eidx, fidx = ifaceL[0:3]
                self._int_eidxs[etype,fidx].append(eidx)
                self._eidxs[etype,fidx].append(eidx)
                self._intinters.append([ifaceL[0:3],ifaceR[0:3]])
                self._inters.append([etype,fidx,eidx])
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)
                #add both right and left faces for vtu writing
                etype, eidx, fidx = ifaceR[0:3]
                self._int_eidxs[etype,fidx].append(eidx)
                self._eidxs[etype,fidx].append(eidx)
                self._intinters.append([ifaceR[0:3],ifaceL[0:3]])
                self._inters.append([etype,fidx,eidx]) 
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)

        for (etype,fidx) in self._int_eidxs:
            self._nintinters[etype,fidx] = len(self._int_eidxs[etype,fidx])
            #self._int_eidxs[etype,fidx] = np.moveaxis(np.array(self._int_eidxs[etype,fidx]),0,1)
            
    def collect_mpiinters(self,rallocs,mesh,eset):
        comm, rank, root = get_comm_rank_root()
        prank = rallocs.prank
        # send flags
        for rhs_prank in rallocs.prankconn[prank]:
            conkey = f'con_p{prank}p{rhs_prank}'
            mpiint = mesh[conkey].astype('U4,i4,i1,i2').tolist()
            flagL = np.zeros((len(mpiint)),dtype=bool)
            for findex, ifaceL in enumerate(mpiint):
                etype, eidx, fidx = ifaceL[0:3]
                flagL[findex] = (eidx in eset[etype]) if etype in eset else False
            rhs_mrank = rallocs.pmrankmap[rhs_prank]
            comm.Send(flagL,rhs_mrank,tag=52)

        # receive flags and collect mpi interfaces
        self.mpi_scountmap = {}
        self.mpi_rcountmap = {}
        for rhs_prank in rallocs.prankconn[prank]:
            conkey = f'con_p{prank}p{rhs_prank}'
            mpiint = mesh[conkey].astype('U4,i4,i1,i2').tolist()
            flagR = np.empty((len(mpiint)),dtype=bool)
            rhs_mrank = rallocs.pmrankmap[rhs_prank] 
            comm.Recv(flagR,rhs_mrank,tag=52)

            sc = 0
            rc = 0
            for findex, ifaceL in enumerate(mpiint):
                etype, eidx, fidx = ifaceL[0:3]
                flagL = (eidx in eset[etype]) if etype in eset else False
                # add info if it is an fwh edge
                if flagL and not flagR[findex] :
                    self._eidxs[etype,fidx].append(eidx)
                    self._mpi_eidxs[etype,fidx].append(eidx)
                    self._inters.append([etype,fidx,eidx])
                    if eidx not in  self._inside_eset[etype]:
                        self._inside_eset[etype].append(eidx)
                    rc += 1
                elif flagR[findex] and not flagL:
                    sc += 1

                #periodic and mpi interface
                elif (flagL and flagR[findex]) and ifaceL[-1] !=0 :
                    self._eidxs[etype,fidx].append(eidx)
                    self._mpi_eidxs[etype,fidx].append(eidx)
                    self._inters.append([etype,fidx,eidx])
                    if eidx not in  self._inside_eset[etype]:
                        self._inside_eset[etype].append(eidx)
                    rc += 1
                    sc += 1

            if rc or sc: # this means there is a fwh mpi interface
                if rhs_mrank not in self.fwh_edgranks_list:
                    self.fwh_edgranks_list.append(rhs_mrank)
                if rank not in self.fwh_edgranks_list:
                    self.fwh_edgranks_list.append(rank)
            self.mpi_scountmap[rhs_mrank] = sc
            self.mpi_rcountmap[rhs_mrank] = rc
        #do not send to yourself:
        self.mpi_scountmap[rank] = 0
        self.mpi_rcountmap[rank] = 0

        for (etype,fidx) in self._mpi_eidxs:
            self._nmpiinters[etype,fidx] = len(self._mpi_eidxs[etype,fidx])

    def collect_bndinters(self,rallocs,mesh,eset):
        prank = rallocs.prank
        for f in mesh:
            if (m := re.match(f'bcon_(.+?)_p{prank}$', f)):
                bname = m.group(1)
                bclhs = mesh[f].astype('U4,i4,i1,i2').tolist()
                for findex, ifaceL in enumerate(bclhs):
                    etype, eidx, fidx = ifaceL[0:3]
                    flagL = (eidx in eset[etype]) if etype in eset else False
                    if flagL:
                        self._bndinters[etype,fidx].append((eidx,findex,bname))
                        self._bnd_eidxs[etype,fidx].append(eidx)
                        self._eidxs[etype,fidx].append(eidx)
                        self._inters.append([etype,fidx,eidx])
                        if eidx not in  self._inside_eset[etype]:
                            self._inside_eset[etype].append(eidx)

        for (etype,fidx) in self._bnd_eidxs:
            self._nbndinters[etype,fidx] = len(self._bnd_eidxs[etype,fidx])

    def build_fwh_subranks_lists(self,intg):
        self.fwhranks_list = []
        comm, rank, root = get_comm_rank_root()
        
        # rallocs = intg.rallocs
        # prank = rallocs.prank
        # if comm.size > 1:
        #     pp= []
        #     mm = [] 
        #     for rhs_prank in rallocs.prankconn[prank]:
        #         pp.append(rhs_prank)
        #         mm.append(rallocs.pmrankmap[rhs_prank])
        #     mranks_, pranks_ = (list(t) for t in zip(*sorted(zip(mm, pp))))
    
        #     if self._mpi_eidxs:
        #         rallocs = intg.rallocs
        #         prank = rallocs.prank
        #         print(f'\n=================================\nmrank {rank} --> prank {prank} \n=================================')
        #         print(f'FwhEdgeRanks: {self.fwh_edgranks_list}', flush=True)
        #         mranks = []
        #         for rhs_prank in rallocs.prankconn[prank]:
        #             mranks.append(rallocs.pmrankmap[rhs_prank])
        #         print(f'rhs mranks {mranks} --> pranks {rallocs.prankconn[prank]}')
        #         print(f'rhs mranks {mranks_} --> pranks {pranks_}  --- sorted')
            
        #         for (etype,fidx) in self._mpi_eidxs:
        #             print(f'  nmpiinters[{etype},{fidx}]: {self._nmpiinters[etype,fidx]}')
        # sys.stdout.flush()
        # Determine active ranks for fwh computations
        self.active_fwhrank = True if self._eidxs else False
        rank_is_active_list = comm.allgather(self.active_fwhrank)
        for i, flag in enumerate(rank_is_active_list):
            if flag:
                self.fwhranks_list.append(i)

        # Determine fwh mpi edge/interface sharing ranks
        count = comm.allgather(len(self.fwh_edgranks_list))
        recvbuf = np.zeros(sum(count),dtype='i')
        displ = [sum(count[:p]) for p in range(len(count))]
        displ = np.array(displ)
        self.fwh_edgranks_list = np.array(self.fwh_edgranks_list, dtype='i')

        comm.Allgatherv(self.fwh_edgranks_list,[recvbuf, count, displ, MPI.INT])
        self.fwh_edgranks_list = list(set(np.sort(recvbuf)))
        self.active_fwhedgrank = True if rank in self.fwh_edgranks_list else False
        
        # Constructing sub-communicators and sub-groups
        fwh_group = comm.group.Incl(self.fwhranks_list)
        fwh_edgegroup = comm.group.Incl(self.fwh_edgranks_list)
        fwh_comm = comm.Create(fwh_group)
        fwh_edgecomm = comm.Create(fwh_edgegroup)
        if rank not in self.fwhranks_list:
            fwh_group = MPI.GROUP_NULL
            fwh_comm = MPI.COMM_NULL
        if rank not in self.fwh_edgranks_list:
            fwh_edgegroup = MPI.GROUP_NULL
            fwh_edgecomm = MPI.COMM_NULL

        if fwh_comm is not MPI.COMM_NULL:
            self.fwhrank = fwh_comm.rank
        if fwh_edgecomm is not MPI.COMM_NULL:
            self.fwhedgrank = fwh_edgecomm.rank

        self.mpiedge_scount = [0] * len(self.fwh_edgranks_list)
        self.mpiedge_rcount = [0] * len(self.fwh_edgranks_list)
        
        for ir, rk in enumerate(self.fwh_edgranks_list):
            if rk in self.mpi_scountmap:
                self.mpiedge_scount[ir] = self.mpi_scountmap[rk]
                self.mpiedge_rcount[ir] = self.mpi_rcountmap[rk]
        
        # comm.barrier()
        # if self.active_fwhrank and self.active_fwhedgrank:
        #     print(f'ranks ({rank},{prank},f{self.fwhrank},e{self.fwhedgrank}), scount {self.mpiedge_scount}, rcount {self.mpiedge_rcount}', flush=True)
        # elif self.active_fwhedgrank:
        #     print(f'ranks ({rank},{prank},e{self.fwhedgrank}), scount {self.mpiedge_scount}, rcount {self.mpiedge_rcount}', flush=True)
        # comm.barrier()
        # sys.stdout.flush()
        
        if rank == root:
            #print(f'\nnum of fwh surface ranks {len(self.fwhranks_list)}')
            print(f'\n{len(self.fwhranks_list)} fwh surface ranks: {self.fwhranks_list}')
            #print(f'\nnum of fwh mpi/edge  ranks {len(self.fwh_edgranks_list)}')
            print(f'{len(self.fwh_edgranks_list)} fwh mpi/edge ranks: {self.fwh_edgranks_list}\n')
        sys.stdout.flush()

        return fwh_comm, fwh_edgecomm


    def get_nqpts_m0(self, elemap, eidxs):
        m0 = {}
        nqpts = 0
        for (etype,fidx), eidlist in eidxs.items():
            eles = elemap[etype]
            if (etype, fidx) not in m0:
                facefpts = eles.basis.facefpts[fidx]
                m0[etype,fidx] = eles.basis.m0[facefpts]
                nfpts_pertype, _ = m0[etype,fidx].shape
            nfaces_pertype = len(eidlist)
            nqpts += (nfpts_pertype * nfaces_pertype)
        return nqpts, m0

    def get_fwhsurf_finfo(self, elemap, eidxs):

        m0 = {}
        srtd_ids = defaultdict(list)
        ndims = self.ndims
        nqpts = 0
        nfaces = 0

        for (etype,fidx), eidlist in eidxs.items():
            eles = elemap[etype]
            
            if (etype, fidx) not in m0:
                facefpts = eles.basis.facefpts[fidx]
                m0[etype,fidx] = eles.basis.m0[facefpts]
                qwts = eles.basis.fpts_wts[facefpts]
                nfpts_pertype, _ = m0[etype,fidx].shape

            nfaces_pertype = len(eidlist)
            mpn = np.empty((nfaces_pertype,nfpts_pertype))
            npn = np.empty((nfaces_pertype,nfpts_pertype,ndims))
            fplocs = np.empty((nfaces_pertype,nfpts_pertype,ndims))
            
            for ie, eidx in enumerate(eidlist):
                # get sorted fpts ids
                fpts_idx = eles._srtd_face_fpts[fidx][eidx]
                sids = np.argsort(fpts_idx)
                srtd_ids[etype,fidx].append(sids)

                # save flux pts coordinates:
                #fpts[etype, fidx].append(eles.get_ploc_for_inter(eidx,fidx)) # sorted
                fplocs[ie] = eles.plocfpts[facefpts, eidx] # unsorted

                # Unit physical normals and their magnitudes (including |J|)
                mpn[ie] = eles.get_mag_pnorms(eidx, fidx)
                npn[ie] = eles.get_norm_pnorms(eidx, fidx)
            
            qdA = np.einsum('i,ji->ji',qwts,mpn)
            
            dA = qdA.reshape(-1) if nqpts == 0  else np.hstack((dA,qdA.reshape(-1)))
            fpts_plocs = fplocs.reshape(-1,ndims)  if nqpts == 0 else np.vstack((fpts_plocs,fplocs.reshape(-1,ndims) ))
            norm_pnorms = npn.reshape(-1,ndims) if nqpts == 0  else np.vstack((norm_pnorms,npn.reshape(-1,ndims) ))

            nqpts += (nfpts_pertype * nfaces_pertype)
            nfaces += nfaces_pertype

        qinfo = [fpts_plocs,norm_pnorms,dA] if m0 else [[],[],[]]
        
        #debugging
        comm, rank, root = get_comm_rank_root()
        shape0 = comm.reduce(np.asarray(qinfo[0]).shape[0],root=0) 
        shape1 = comm.reduce(np.asarray(qinfo[1]).shape[0],root=0) 
        shape2 = comm.reduce(np.asarray(qinfo[2]).shape[0],root=0) 
        if rank == root:
            print(f'surfinfo.shapes: fpts {shape0} , norms {shape1}, qdA {shape2}')
        nfaces_tot = comm.reduce(nfaces,root=0)
        nqpts_tot = comm.reduce(nqpts,root=0)
        if rank == root:
            print(f'nfaces : {nfaces_tot}')
            print(f'nqpts  : {nqpts_tot}')
        sys.stdout.flush() 

        return qinfo,m0    

    def update_usoln_from_currstep(self, intg, m0_dict, eidxs, ushape):

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # fwh soln array for one time step:
        usoln = np.empty((ushape))
        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))
        ndims, nvars = self.ndims, self.nvars

        fIo = 0
        for (etype, fidx), m0 in m0_dict.items(): 
            # Get the interpolation operator
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][...,eidxs[etype,fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1) # nvars, nfpts, nfaces
            #ufpts = np.swapaxes(ufpts.reshape(nvars,-1)[...,srtids_],1,2)  # additional sorting

            # get primitive vars
            pri_ufpts = self.elementscls.con_to_pri(ufpts, self.cfg)
            fIsize = pri_ufpts.shape[1] * pri_ufpts.shape[2]
            fImax = fIo + fIsize
            if self._artificial_compress:
                usoln[-1,fIo:fImax] = pp = pri_ufpts[0].reshape(-1)  #p
                usoln[0,fIo:fImax] = pp/self.constvars['ac-zeta'] #rho
            else:
                usoln[-1,fIo:fImax] = pri_ufpts[-1].reshape(-1)
                usoln[0,fIo:fImax] = pri_ufpts[0].reshape(-1)
            usoln[1,fIo:fImax] = pri_ufpts[1].reshape(-1)
            usoln[2,fIo:fImax] = pri_ufpts[2].reshape(-1)
            if ndims == 3:
                usoln[3,fIo:fImax] = pri_ufpts[3].reshape(-1) 
            fIo = fImax
        
        return usoln
        

class FFTParam(object):
    #if more windows are needed, they can be customally added. A larger list is available in scipy
    windows = {
        'None'    : (lambda s: np.ones(s),      {'density': 1., 'spectrum': 1.}),  # rectangle window
        'hanning' : (lambda s: np.hanning(s),   {'density': np.sqrt(8./3.), 'spectrum': 2.}),
        'hamming' : (lambda s: np.hamming(s),   {'density': 50.*np.sqrt(3974.)/1987., 'spectrum': 50./27.}),
        'blackman': (lambda s: np.blackman(s),  {'density': 50.*np.sqrt(3046.)/1523., 'spectrum': 50./21.}),
        'bartlett': (lambda s: np.bartlett(s),  {'density': np.sqrt(3.), 'spectrum': 2.}),
        }

    def __init__(self,dt_sim,dt_sub,ltsub,shift,window_func,psd_scale_mode):
        self.dt_sim = dt_sim
        self.dt_sub = dt_sub
        self.ltsub =  ltsub       
        self.shift = shift
        self.window_func = window_func
        self.psd_scale_mode = psd_scale_mode
        self.tol = 1e-12

        if (self.ltsub + self.tol) <= self.dt_sim:
            raise ValueError(f'fft param, ltsub window length {self.ltsub} is too short or less than simulation time step {self.dt_sim}')
        if self.shift > 1.:
            raise  ValueError(f'window overlap/shift {self.shift} cannot exceed one, please adjust it as necessary')

        self.prepare_fft_param()

    def prepare_fft_param(self):
        ltsub = self.ltsub
        dt_sub = self.dt_sub
        ntsub = ltsub/dt_sub 
        shift = self.shift
        window = self.window_func
        scaling_mode = self.psd_scale_mode
        dt_sim = self.dt_sim

        # Adjust inputs
        #(1) dt_sub
        self.dt_sub = dt_sub = int(np.rint(dt_sub/dt_sim))*dt_sim  if dt_sub > dt_sim else dt_sim
        #(2) compute ntsub as a power of 2 number
        N = int(np.rint(ltsub/dt_sub))
        self.ntsub = ntsub = get_num_nearest_pow_of_two(N)
        #self.ntsub = ntsub = N
        #(3): adjust the window length
        self.ltsub = ltsub = ntsub * dt_sub

        #(4) Adjusting shift parameters for window averaging and overlapping
        # partial (0.01 < shift < 1) or complete overlap (shift = 1)
        if shift > 0.01:
            self.aver_count = 0
            self.averaging = True
            self.Nt_shifted = int(np.rint(ntsub*shift))
        # no averaging or shifting, shift < 0.01 
        else:
            self.aver_count = 0
            self.averaging= False
            self.Nt_shifted = ntsub    
        self.Fsampl = 1./self.dt_sim
        self.fsampl_sub = 1./self.dt_sub
        self.steps_count = 0
        self.samplesteps = self.dt_sub/self.dt_sim
        
        #(5) window function params
        #since we are using windows for spectral analysis we do not use the last data entry
        self.wwind = self.windows[window][0](ntsub+1)[:-1]
        self.windscale = self.windows[window][1][scaling_mode]

        if get_comm_rank_root()[1] == 0:
            print(f'\n--------------------------------------')
            print(f'       Adjusted FFT parameters ')
            print(f'--------------------------------------')
            print(f'sample freq : {self.fsampl_sub} Hz')
            print(f'delta freq  : {1./self.ltsub} Hz')
            print(f'dt window   : {self.dt_sub} sec')
            print(f'Lt window   : {self.ltsub} sec')
            print(f'Nt window   : {self.ntsub}')
            print(f'Nt shifted  : {self.Nt_shifted}')
            if self.averaging:
                print(f'PSD Averaging is \'activated\'')
            else:
                print(f'PSD Averaging is \'not activated\'')
            print(f'window function is \'{self.window_func}\'')
            print(f'psd scaling mode is \'{self.psd_scale_mode}\'\n')


class FwhSolver(object):
    #reference pressure for spl computation
    pref = 2e-5

    def __init__(self,observers,surfdata,Uinf,fftcls,surftype='permeable'):
        self.nobserv = len(observers) # number of observers
        self.obsvlocs = observers
        self.uinf = Uinf
        self.fftcls = fftcls
        self.surfdata = surfdata
        self.xyz_src, self.qnorms, self.qdA = self.surfdata
        self.nvars = len(self.uinf['u'])+2
        self.nqpts = np.asarray(self.surfdata[0]).shape[0]
        self.ndims = len(observers[0]) 

        # compute distance vars for fwh
        self.magR, self.magRs, self.R_nvec, self.Rs_nvec = self._compute_distance_vars(self.xyz_src,self.obsvlocs,self.uinf['Mach'])
        # compute frequency parameters
        self._freq  = np.fft.rfftfreq(self.fftcls.ntsub,self.fftcls.dt_sub)
        self.omega = 2*np.pi*self._freq
        self.kwv = self.omega/self.uinf['c']
        self.nfreq = np.size(self._freq)

        # Init fwh outputs 
        self._pfft = np.zeros((self.nobserv,self.nfreq),dtype=np.complex128)
        self._pmagsum = np.zeros((self.nobserv,self.nfreq))
        self._presum = np.zeros((self.nobserv,self.nfreq))
        self._pimgsum = np.zeros((self.nobserv,self.nfreq))

        self.solidsurf = True if surftype == 'solid' else False

        self.__usoln = self._allocate_usoln_darray(self.ntsub, self.nvars, self.nqpts)

    def _allocate_usoln_darray(self, ntsub, nvars, nqpts):
        return np.empty((ntsub,nvars,nqpts))

    @staticmethod
    def _compute_distance_vars(xyz_src,xyz_ob,Minf):
        nobserv = xyz_ob.shape[0]
        ndims = xyz_ob[0].shape[0]
        magMinf = np.sqrt(sum(m*m for m in Minf)) # |M|
        # inverse of Prandtl Glauert parameter
        gm_ = 1. / np.sqrt(1.0 - magMinf*magMinf) 
        gm2_ = gm_*gm_
        # dr = x_observer - x_source
        dr = np.array([ob - xyz_src for ob in xyz_ob]).reshape(-1,ndims)    
        magdr = np.sqrt(sum((dr*dr).T)) # distance |dr|
        ndr = np.array([idr/ir for idr,ir in zip(dr,magdr)])  # normalized unit dr
        Minf_ndr = sum((Minf*ndr).T) # Minfty.r_hat
        Rs = (magdr/gm_) * np.sqrt(1.+ gm2_*Minf_ndr*Minf_ndr)   # |R*|
        R = gm2_ * (Rs - magdr*Minf_ndr)               # |R|
        Minf2_ndr = np.einsum('i,j->ji',Minf,Minf_ndr)
        rrs = magdr/(gm2_*Rs)
        mr = ndr + Minf2_ndr*gm2_
        nRs = np.einsum('i,ij->ij',rrs,mr) # normalized unit R*
        nR = gm2_ * (nRs-Minf)    # normalized unit R
        if not (R.shape[0] == nobserv):
            nR = nR.reshape(nobserv,-1,ndims)
            nRs = nRs.reshape(nobserv,-1,ndims)
            R = R.reshape(nobserv,-1)
            Rs = Rs.reshape(nobserv,-1)
 
        return R,Rs,nR,nRs

    def compute_fwh_solution(self):
        psd_scale_mode = self.fftcls.psd_scale_mode
        pmagsum, presum, pimgsum = self._pmagsum, self._presum, self._pimgsum
        #compute fluxes in time domain
        Q, F = self._compute_surface_fluxes(self.__usoln)
        #compute fwh pressure solution in frequency domain
        self._pfft = pfft = self._compute_observer_pessure(Q,F)
        # use welch method if averaging
        if self.fftcls.averaging:
            pmagsum, presum, pimgsum = welch_accum(pfft,pmagsum,presum,pimgsum)
            if self.fftcls.aver_count > 0:
                self._pfft = welch_average(self.naver,pmagsum,presum,pimgsum)
            self._pmagsum, self._presum, self._pimgsum = pmagsum, presum, pimgsum
            #update the averaging counter
            self.fftcls.aver_count += 1
        #update the step counter
        self.fftcls.steps_count = self.fftcls.ntsub - self.fftcls.Nt_shifted

    def _compute_surface_fluxes(self, usoln):
        # define local vars
        qnorms = self.qnorms
        cinf = self.uinf['c']
        pinf = self.uinf['p']
        rhoinf = self.uinf['rho']
        uinf = np.array(self.uinf['u'])
        # compute soln vars
        u = np.swapaxes(usoln[:,1:self.ndims+1,:],1,2)-uinf # u' perturbed vel
        p = usoln[:,-1,:] - pinf # p' fluctuation pressure
        rho_tot = rhoinf + p/(cinf*cinf) # density, another formulation will just use rho in usoln[:,0,:]
        # compute normal velocities
        un  = np.einsum('ij,kij->ki',qnorms,u) # normal flow velocity
        Uin = sum((uinf*qnorms).T)  # normal Uinfty
        vn = un + Uin if self.solidsurf else 0.
        dvn = vn - Uin  # relative surface velocity
        dUn = un - dvn  # relative normal flow velocity
        # compute temporal fluxes
        Q = rho_tot*dUn + rhoinf*dvn
        rdun = rho_tot*dUn
        umuinf = u-uinf
        rhouinf = rhoinf*uinf
        F = np.einsum('ij,ijk->ijk',rdun,umuinf)
        F += np.einsum('i,j...->j...i',rhouinf,dvn)
        F += np.einsum('ij,jk->ijk',p,qnorms)

        return Q,F
        
    def _compute_observer_pessure(self,Q,F):
        Rv_ = self.R_nvec
        Rsv_ = self.Rs_nvec
        magR = self.magR
        magRs = self.magRs
        wwind =self.fftcls.wwind
        qdA = self.qdA
        
        # fluxes signal windowing
        if self.fftcls.window_func != 'none':
            Q -= np.mean(Q,0)
            Q = np.einsum('i,ij->ij',wwind,Q)
            F -= np.mean(F,0)
            F = np.einsum('i,ijk->ijk',wwind,F)

        # perform fft of Q, F fluxes
        Qfft = np.empty((self.nfreq,self.nqpts),dtype=np.complex64)
        Ffft = np.empty((self.nfreq,self.nqpts,self.ndims),dtype=np.complex64)
        for iq in range(0,self.nqpts):
            Qfft[:,iq] = rfft(Q[:,iq])
            for jd in range(0,self.ndims):
                Ffft[:,iq,jd]  = rfft(F[:,iq,jd])

        #compute pfft, i=nob,j=nfreq,k=nqpts, p shape i,j,k 
        kwvR = np.einsum('ik,j->jik',magR,self.kwv)
        exp_term0 = np.exp(-1j*kwvR)        # exp(-ikR)
        exp_term1 = exp_term0/magRs          # exp(-ikR)/R*
        exp_term2 = exp_term0/(magRs*magRs) # exp(-ikR)/(R*xR*)
        pfft  = 1j * np.einsum('j,jik,jk->ijk',self.omega, exp_term1, Qfft)        # p1_term
        pfft += 1j * np.einsum('j,jik,ikm,jkm->ijk',self.kwv,exp_term1,Rv_,Ffft)   # p2_term0
        pfft += np.einsum('jik,ikm,jkm->ijk',exp_term2,Rsv_,Ffft)                  # p2_term1 

        # surface integration
        pfft = self._surface_integrate(qdA,pfft)
        pfft *= self.fftcls.windscale/(4.*np.pi)

        return pfft

    def _surface_integrate(self,qdA,p):
        #i=nobservers, j=nfreq or ntime, k=nqpts
        return np.einsum('ijk,k->ij',p,qdA)

    def check_compute(self):
        return (self.nsteps >= self.ntsub)

    @property
    def nsteps(self):
        return self.fftcls.steps_count
    @property
    def naver(self):
        return self.fftcls.aver_count
    @cached_property
    def ntsub(self):
        return self.fftcls.ntsub
    @cached_property
    def averaging(self):
        return self.fftcls.averaging
    @cached_property
    def window_func(self):
        return self.fftcls.window_func
    @cached_property
    def freq(self):
        return self._freq
    @property
    def pfft(self):
        return self._pfft

    @property
    def pmagsum(self):
        return self._pmagsum
    @pmagsum.setter
    def pmagsum(self,val):
        self._pmagsum = val

    @property
    def presum(self):
        return self._presum
    @presum.setter
    def presum(self,val):
        self._presum = val

    @property
    def pimgsum(self):
        return self._pimgsum
    @pimgsum.setter
    def pimgsum(self,val):
        self._pimgsum = val

    @property
    def usoln_onestep(self):
         return self.__usoln[self.fftcls.steps_count]

    @usoln_onestep.setter
    def usoln_onestep(self,val):
        self.__usoln[self.fftcls.steps_count] = val
        self.fftcls.steps_count += 1

    def __call__(self, *args, **kwds):
        pass


class PointAcousticSrc(object):
    def __init__(self,name,srclocs,flowinfo,*srcdata):
        self.name = name
        self.srclocs = srclocs
        self.ndims = len(srclocs)
        self.nvars = len(flowinfo)
        self.rhoinf, self.uinf, self.pinf = flowinfo[0], flowinfo[1:self.ndims+1], flowinfo[-1]

    def exact_solution(self,*args):
        pass
    def update_usoln_onestep(self,*args):
        pass

class MonopoleSrc(PointAcousticSrc):
    def __init__(self,name,srclocs,flowinfo,ampl,srcfreq,ntime,nperiods,gamma):
        super().__init__(name,srclocs,flowinfo,ampl,srcfreq,ntime,nperiods,gamma)
        self.gamma = gamma
        self.ampl = ampl
        self.srcfreq = srcfreq
        self.nperiods = nperiods
        self.nt = ntime
        self.omega = 2.*np.pi*self.srcfreq
        self.tperiod = 2.*np.pi/self.omega
        self.dt = self.nperiods*self.tperiod/self.nt
        self.freq = np.fft.rfftfreq(self.nt,self.dt)
        self.nfreq = len(self.freq)

        co = np.sqrt(self.gamma*self.pinf/self.rhoinf) 
        print(f'src locs {self.srclocs}')
        print(f'co {co}, rho_o {self.rhoinf}, po {self.pinf}')
        print(f'Minf {self.uinf/co}')
        sys.stdout.flush()
        
    def exact_solution(self,xyz_ob):
        nobserv = xyz_ob.shape[0]
        ptime = np.empty((self.nt,nobserv))
        pfft = np.empty((nobserv,self.nfreq),dtype=np.complex128)
        for i in range(self.nt):
            ptime[i] = self.update_usoln_onestep(xyz_ob,i*self.dt,nobserv,'p') - self.pinf
        ptime = np.moveaxis(ptime,0,1)
        for i in range(nobserv):
            pfft[i] = rfft(ptime[i])
        return self.freq,pfft

    def update_usoln_onestep(self,xyz_ob,tcurr,ushape,uout='usoln'):
        usoln = np.empty(ushape)
        co = np.sqrt(self.gamma*self.pinf/self.rhoinf)
        Mo = self.uinf/co
        kwv = self.omega/co
        magR, magRs, R_nvec, Rs_nvec = FwhSolver._compute_distance_vars(self.srclocs,xyz_ob,Mo)
        # phy_potential of the source and its derivatives
        phy = (self.ampl/(4.*np.pi*magRs))*np.exp(1j*(self.omega*tcurr-kwv*magR)) 
        invRs = 1./magRs
        kk = 1j*kwv*R_nvec + np.einsum('i,ij->ij',invRs,Rs_nvec)
        dphy = - np.einsum('i,ij->ji',phy,kk) 
        mdotdphy = sum((Mo*dphy.T).T)

        # computing the flow quantities
        # pressure
        p = np.real(-self.rhoinf * (1j*self.omega*phy + co * mdotdphy)) 
        if not (uout == 'usoln'):
            return p + self.pinf
        
        usoln[-1] = p + self.pinf
        usoln[0] = p/(co*co) + self.rhoinf  #rho
        usoln[1:self.ndims+1] = usrc = np.array([ui + uo for ui, uo in zip(np.real(dphy),self.uinf)])  #u
        
        return usoln


class DipoleSrc(PointAcousticSrc):
    def __init__(self,name,srclocs,flowinfo,*srcdata):
        super().__init__(name,srclocs,flowinfo,*srcdata)

class VTUSurfWriter(object):

    vtkfile_version = '2.1'
    vtk_types = dict(line=3, tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    vtk_nodes = dict(tri=3, quad=4, tet=4, pyr=5, pri=6, hex=8)
    # number of first order nodes/faces per element
    _petype_focount_map = {'line': 2, 'tri': 3, 'quad': 4,
                       'tet': 4, 'pyr': 5, 'pri': 6, 'hex': 8} 
    # map of first order face nodes
    # To generate this fnmap, we use vtk_nodemaps and then apply gmsh_fnmap 
    # to extract the correct first order face node ordering.
    # Some fnmap face points may need to be flipped to maintain a c.c.w or c.c. 
    # face node counting
    _petype_fnmap = {
        ('tri',  3 ):  {'line': [[0, 1], [1, 2], [2, 0]]},
        ('tri',  6 ):  {'line': [[5, 0], [0, 2], [2, 5]]},
        ('tri',  10 ):  {'line': [[9, 0], [0, 3], [3, 9]]},
        ('tri',  15 ):  {'line': [[14, 0], [0, 4], [4, 14]]},
        ('quad',  4 ):  {'line': [[0, 1], [1, 3], [3, 2], [2, 0]]},
        ('quad',  9 ):  {'line': [[0, 2], [2, 8], [8, 6], [6, 0]]},
        ('quad',  16 ):  {'line': [[0, 3], [3, 15], [15, 12], [12, 0]]},
        ('quad',  25 ):  {'line': [[0, 4], [4, 24], [24, 20], [20, 0]]},
        ('tet',  4 ):  {'tri': [[1, 0, 3], [3, 0, 2], [2, 1, 3], [0, 1, 2]]},
        ('tet',  10 ):  {'tri': [[2, 0, 9], [9, 0, 5], [5, 2, 9], [0, 2, 5]]},
        ('tet',  20 ):  {'tri': [[3, 0, 19], [19, 0, 9], [9, 3, 19], [0, 3, 9]]},
        ('tet',  35 ):  {'tri': [[4, 0, 34], [34, 0, 14], [14, 4, 34], [0, 4, 14]]},
        ('hex',  8 ):  {'quad': [[0, 1, 3, 2], [0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7], [0, 2, 6, 4], [4, 5, 7, 6]]},
        ('hex',  27 ):  {'quad': [[0, 2, 8, 6], [0, 2, 20, 18], [2, 8, 26, 20], [8, 6, 24, 26], [0, 6, 24, 18], [18, 20, 26, 24]]},
        ('hex',  64 ):  {'quad': [[0, 3, 15, 12], [0, 3, 51, 48], [3, 15, 63, 51], [15, 12, 60, 63], [0, 12, 60, 48], [48, 51, 63, 60]]},
        ('hex',  125 ):  {'quad': [[0, 4, 24, 20], [0, 4, 104, 100], [4, 24, 124, 104], [24, 20, 120, 124], [0, 20, 120, 100], [100, 104, 124, 120]]},
        ('pri',  6 ):  {'quad': [[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]], 'tri': [[0, 2, 1], [3, 4, 5]]},
        ('pri',  18 ):  {'quad': [[0, 2, 14, 12], [2, 5, 17, 14], [0, 12, 17, 5]], 'tri': [[0, 5, 2], [12, 14, 17]]},
        ('pri',  40 ):  {'quad': [[0, 3, 33, 30], [3, 9, 39, 33], [0, 30, 39, 9]], 'tri': [[0, 9, 3], [30, 33, 39]]},
        ('pri',  75 ):  {'quad': [[0, 4, 64, 60], [4, 14, 74, 64], [0, 60, 74, 14]], 'tri': [[0, 14, 4], [60, 64, 74]]},
        ('pyr',  5 ):  {'quad': [[2, 3, 1, 0]], 'tri': [[0, 1, 4], [1, 3, 4], [3, 2, 4], [0, 4, 2]]},
        ('pyr',  14 ):  {'quad': [[6, 8, 2, 0]], 'tri': [[0, 2, 13], [2, 8, 13], [8, 6, 13], [0, 13, 6]]},
        ('pyr',  30 ):  {'quad': [[12, 15, 3, 0]], 'tri': [[0, 3, 29], [3, 15, 29], [15, 12, 29], [0, 29, 12]]},
        ('pyr',  55 ):  {'quad': [[20, 24, 4, 0]], 'tri': [[0, 4, 54], [4, 24, 54], [24, 20, 54], [0, 54, 20]]},
    }
    # offset to get local fidx/fnum inside each facetype map
    _fnum_offset = {'tri' : {'line': 0},
                    'quad': {'line': 0},
                    'tet' : {'tri': 0},
                    'hex' : {'quad': 0},
                    'pri' : {'quad': -2, 'tri': 0}, 
                    'pyr' : {'quad': 0, 'tri': -1}}
    # reverse map from face index to face type
    fnum_pftype_map = {
            'tri' : { 0: 'line', 1: 'line', 2: 'line'}, 
            'quad': { 0: 'line', 1: 'line', 2: 'line', 3: 'line'}, 
            'tet' : { 0: 'tri', 1: 'tri', 2: 'tri', 3: 'tri'}, 
            'hex' : { 0: 'quad', 1: 'quad', 2: 'quad', 3: 'quad', 4: 'quad', 5: 'quad'},
            'pri' : { 0: 'tri', 1: 'tri', 2: 'quad', 3: 'quad', 4: 'quad'}, 
            'pyr' : { 0: 'quad', 1: 'tri', 2: 'tri', 3: 'tri', 4: 'tri'} 
    }

    _vtufnodes = defaultdict(list) # fwh face to vtu nodes set

    def __init__(self, intg, eidxs, fieldvars=None, fielddata=None):
        self.mesh = intg.system.mesh
        self.rallocs = intg.rallocs
        self._eidxs = eidxs
        self.ndims = intg.system.ndims
        # Output data type
        self.fpdtype = intg.backend.fpdtype

        self._vtk_vars = fieldvars.extend(('Partition', 'r')) if fieldvars else [('Partition', 'r')]
        self._vtk_fields = fielddata if fielddata else []

        self._prepare_vtufnodes()

    def write(self,fname):
        fname = f'{fname}.vtu'
        self._write_vtu_out(fname)

    def _prepare_vtufnodes(self): 
        for (etype,fidx), eidxlist in self._eidxs.items():
            pts = np.swapaxes(self.mesh[f'spt_{etype}_p{self.rallocs.prank}'],0,1)
            pftype = self.fnum_pftype_map[etype][fidx]
            fidx = np.int(fidx) + self._fnum_offset[etype][pftype]
            for eidx in eidxlist:
                #eidx = np.int(eidx)
                nelemnodes = pts[eidx].shape[0]
                nidx = self._petype_fnmap[etype,nelemnodes][pftype][fidx]
                self._vtufnodes[pftype].append(pts[eidx][nidx,:])

    def _prepare_vtufnodes_info(self):
        self.partsdata = {}
        comm, rank, root = get_comm_rank_root()
        info = defaultdict(dict)
        for k, v in self._vtufnodes.items():
            npts, ncells, names, types, comps, sizes = self._get_array_attrs(k)
            info[k]['vtu_attr'] = [names, types, comps, sizes]
            info[k]['mesh_attr'] = [npts, ncells]
            info[k]['shape'] = np.asarray(v).shape 
            info[k]['dtype'] = np.asarray(v).dtype.str
            #fileddata
            psize = info[k]['shape'][0]*info[k]['shape'][1]
            self.partsdata[k] = np.tile(rank, psize) 
            info[k]['parts_shape'] = self.partsdata[k].shape
            info[k]['parts_dtype'] = self.partsdata[k].dtype.str
        return info

    def _write_vtu_out(self,fname):

        comm, rank, root = get_comm_rank_root()
        # prepare nodes info for each rank
        info = self._prepare_vtufnodes_info()

        # Communicate and prepare data for writing
        if rank != root:
            # Send the info about our data points to the root rank
            comm.gather(info, root=root)
            # Send the data points itself
            for etype, arrs in self._vtufnodes.items():
                comm.Send(np.array(arrs).astype(info[etype]['dtype']), root, tag=52)
            # Send field data one by one
            for etype in self.partsdata:
                comm.Send(self.partsdata[etype], root, tag=56)
        #root
        else:
            # Collect info about what remote ranks want to write 
            ginfo = comm.gather({}, root)
            # Update the info and receive the node arrays
            vpts_global = {}
            parts_global = {}
            # root data first
            for etype in info:
                vpts_global[etype] = np.array(self._vtufnodes[etype])
                parts_global[etype] = self.partsdata[etype]

            # update info and receive/stack nodes from other ranks
            for mrank, minfo in enumerate(ginfo):
                for etype, vinfo in minfo.items():
                    if etype in info:
                        info[etype]['vtu_attr'][3] = [sum(x) for x in zip(info[etype]['vtu_attr'][3], vinfo['vtu_attr'][3])]
                        info[etype]['mesh_attr'] = [sum(x) for x in zip(info[etype]['mesh_attr'],vinfo['mesh_attr'])]
                        shapes = [x for x in info[etype]['shape']]
                        shapes[0] += vinfo['shape'][0]
                        info[etype]['shape'] = tuple(shapes)
                        pshapes = [x for x in info[etype]['parts_shape']]
                        pshapes[0] += vinfo['parts_shape'][0]
                        info[etype]['parts_shape'] = tuple(pshapes)
                    else:
                        info[etype] = vinfo
                    varr = np.empty(vinfo['shape'], dtype=vinfo['dtype'])
                    comm.Recv(varr, mrank, tag=52)
                    vpts_global[etype] = np.vstack((vpts_global[etype],varr)) if etype in vpts_global else varr

                    parr = np.empty(vinfo['parts_shape'], dtype=vinfo['parts_dtype'])
                    comm.Recv(parr, mrank, tag=56)
                    parts_global[etype] = np.hstack((parts_global[etype],parr)) if etype in parts_global else parr

            # Writing
            write_s_to_fh = lambda s: fh.write(s.encode())

            with open(fname, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                            'byte_order="LittleEndian" '
                            'type="UnstructuredGrid" '
                            f'version="{self.vtkfile_version}">\n'
                            '<UnstructuredGrid>\n')

                # Running byte-offset for appended data
                off = 0
                # Header
                for etype in info:
                    off = self._write_serial_header(fh, info[etype], off)
                write_s_to_fh('</UnstructuredGrid>\n'
                            '<AppendedData encoding="raw">\n_')
                # Data
                for etype in info:
                    self._write_data(fh, etype, vpts_global[etype], parts_global[etype])

                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

        # Wait for the root rank to finish writing
        comm.barrier()

    def _write_darray(self, array, vtuf, dtype):
        array = array.astype(dtype)
        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _process_name(self, name):
        return re.sub(r'\W+', '_', name)

    def _write_serial_header(self, vtuf, info, off):
        names, types, comps, sizes = info['vtu_attr']
        npts, ncells = info['mesh_attr']

        write_s = lambda s: vtuf.write(s.encode())

        write_s(f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n')
        write_s('<Points>\n')

        # Write vtk DaraArray headers
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s(f'<DataArray Name="{self._process_name(n)}" type="{t}" '
                    f'NumberOfComponents="{c}" '
                    f'format="appended" offset="{off}"/>\n')

            off += 4 + s

            # Write ends/starts of vtk file objects
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            elif i == 3:
                write_s('</Cells>\n<PointData>\n')

        # Write end of vtk element data
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_data(self, vtuf, mk, vpts, parts):
        fpdtype = self.fpdtype 
        vpts = vpts.astype(fpdtype)
        neles = vpts.shape[0]
        nfopts = self._petype_focount_map[mk]
        fopts = np.arange(nfopts)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')
        # Write mesh points
        self._write_darray(vpts, vtuf, fpdtype) # simple nodes writer

        # Prepare VTU cell-node connectivity arrays
        vtu_con = np.tile(fopts, (neles, 1))
        vtu_con += (np.arange(neles)*nfopts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(nfopts, (neles, 1))
        vtu_off += (np.arange(neles)*len(fopts))[:, None]

        # Tile VTU cell type numbers
        types = self.vtk_types[mk]
        vtu_typ = np.tile(types, neles)

        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, vtuf, np.int32) 
        self._write_darray(vtu_off, vtuf, np.int32)
        self._write_darray(vtu_typ, vtuf, np.uint8)

        # Writing additional field data:
        self._write_darray(parts, vtuf, np.int32)


    def _get_npts_ncells(self,mk): 
        ncells = np.asarray(self._vtufnodes[mk]).shape[0]
        npts = ncells * self._petype_focount_map[mk]
        return npts, ncells

    def _get_array_attrs(self,mk):
        fpdtype = self.fpdtype 
        vdtype = 'Float32' if fpdtype == np.float32 else 'Float64'
        dsize = np.dtype(fpdtype).itemsize 

        names = ['', 'connectivity', 'offsets', 'types']
        types = [vdtype, 'Int32', 'Int32', 'UInt8']
        comps = ['3', '', '', '']

        vvars = self._vtk_vars
        for fname, varnames in vvars:
            names.append(fname.title())
            types.append('Int32')
            comps.append(str(len(varnames)))

        npts, ncells = self._get_npts_ncells(mk)
        nb = npts*dsize
        sizes = [3*nb, 4*npts, 4*ncells, ncells]
        dsize = np.dtype(np.int32).itemsize
        nb = npts*dsize
        sizes.extend(len(varnames)*nb for _ , varnames in vvars)

        return npts, ncells, names, types, comps, sizes
