# -*- coding: utf-8 -*-

from collections import defaultdict, namedtuple
from mpi4py import MPI
import numpy as np
import os, sys
import re
import time

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.nputil import fuzzysort, npeval
from pyfr.plugins.base import BasePlugin, PostactionMixin
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

def compute_spl(pref, amp, df=None):
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

def compute_psd(amp, df):
    psd = 0.5 * amp * amp / df
    psd[0] *= 2.
    psd[-1] *= 2.
    return psd

def Gatherv_data_arr(comm, rank, root, data_arr):
    recvbuf = None
    displ = None
    count = None
    sbufsize = data_arr.size
    sendbuf = data_arr.reshape(-1)
    count = comm.gather(sbufsize,root=root)

    if rank == root:
        count = np.array(count, dtype='i')
        recvbuf = np.empty(sum(count), dtype=float)
        displ = np.array([sum(count[:p]) for p in range(len(count))])

    comm.Gatherv(sendbuf, [recvbuf, count, displ, MPI.DOUBLE], root=root)

    return recvbuf

def Allgatherv_data_arr(comm, data_arr):
    sbufsize = 0
    sendbuf = np.empty(0)
    if np.any(data_arr):
        sbufsize = data_arr.size 
        sendbuf = data_arr.reshape(-1)
    count = comm.allgather(sbufsize)
    count = np.array(count, dtype='i')
    recvbuf = np.empty(sum(count),dtype=float)
    displ = np.array([sum(count[:p]) for p in range(len(count))])

    comm.Allgatherv(sendbuf, [recvbuf, count, displ, MPI.DOUBLE])

    return recvbuf

def Alltoallv_data_arr(incomm, rhsranks, snddata, ncol, sndcnts, rcvcnts,
                         sndperms, mode='blocking'):
    nsnd = np.asarray(snddata).shape[0]
    sendbuf = np.empty((nsnd, ncol))
    scount = sndcnts
    sdispl = np.array([sum(scount[:p]) for p in range(len(scount))])
    scount = np.array(scount)
    if nsnd:
        for ir, rk in enumerate(rhsranks):
            perm = sndperms[rk]
            sendbuf[sdispl[ir]: sdispl[ir] + scount[ir]] = snddata[perm]
        sendbuf = sendbuf.reshape(-1)
    scount *= ncol
    sdispl *= ncol
    #recv info
    nrcv = np.sum(rcvcnts)
    rbufsize = nrcv * ncol
    recvbuf = np.empty(rbufsize).reshape(-1)
    rcount = np.array(rcvcnts) * ncol
    rdispl = np.array([sum(rcount[:p]) for p in range(len(rcount))])

    s_msg = [sendbuf, scount, sdispl, MPI.DOUBLE]
    r_msg = [recvbuf, rcount, rdispl, MPI.DOUBLE]
    if mode == 'blocking':
        req = incomm.Alltoallv(s_msg, r_msg)
    else:
        req = incomm.Ialltoallv(s_msg, r_msg)
    
    recvbuf = recvbuf.reshape(nrcv, -1) if nrcv else np.empty(0)

    return recvbuf, req


TimeParam = namedtuple('TimeParam', ['dtsim', 'dtsub', 'ltsub', 'shift',
                                     'window', 'psd_scale_mode'])


class FwhSolverPlugin(PostactionMixin, BasePlugin):

    name = 'fwhsolver'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']

    # refernce pressure for spl computation
    pref = 2.e-5
    
    def __init__(self, intg, cfgsect, suffix=None, *args, **kwargs):
        super().__init__(intg, cfgsect, suffix)

        # for restart
        metadata = defaultdict(list)
        metadata = kwargs

        self.DEBUG = self.cfg.getint(cfgsect,'debug',0)
        # Base output directory and file name
        self.basedir = basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        self.basename = self.cfg.get(self.cfgsect, 'basename')
        
        # read time and fft parameters
        self.tstart = self.cfg.getfloat(cfgsect,'tstart',intg.tstart)
        dtsub = self.cfg.getfloat(cfgsect,'dt',intg._dt)
        # read the window length, default is up to simulation tend
        ltsub_def = intg.tend - self.tstart
        ltsub = self.cfg.getfloat(cfgsect,'Lt',ltsub_def)        
        shift = self.cfg.getfloat(cfgsect,'time-shift',0.5)
        window_func = self.cfg.get(cfgsect,'window','hanning')
        psd_scale_mode = self.cfg.get(cfgsect,'psd-scaling-mode','density')
        timeparam = TimeParam(intg._dt,dtsub,ltsub,shift,window_func,psd_scale_mode) 

        # read observers inputs
        self.observers = np.array(self.cfg.getliteral(self.cfgsect, 'observer-locs'))
        self.nobserv = self.observers.shape[0]
        
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
        else:
            self.uinf['rho'] /= self.constvars['gamma']
        self.uinf['c'] = np.sqrt(self.uinf['p']/self.uinf['rho'])
        self.uinf['Mach'] = self.uinf['u']/self.uinf['c']
        self.fwhnvars = len(self.uinf['u'])+2
        
        #surface input data
        #surftype = self.cfg.get(cfgsect,'surftype','permeable')
        #Region of interest
        self.region = region = self.cfg.get(self.cfgsect, 'region')
        # Prepare fwh surface mesh data
        if '(' in region or '[' in region:
            box = self.cfg.getliteral(self.cfgsect, 'region')
            region_eset = self._extract_drum_region_eset(intg, box)
            surftype = 'permeable'
        else:
            region_eset = self._extract_bound_region_eset(intg, region)
            surftype = 'solid'
        self._prepare_surfmesh(intg, region_eset, surftype)

        # Determine fwh active ranks lists
        self.fwh_comm, self.fwh_edgcomm = self._build_fwh_mpi_comms(
                self._eidxs, self._mpi_eidxs, self._mpi_snd_eidxs)   

        # Construct the fwh mesh writer
        self._writer = NativeWriter(intg, basedir, self.basename, 'soln')

        # write the fwh surface geometry file
        self._write_fwh_surface_geo(intg, self._inside_eset, self._eidxs)

        # Underlying elements class
        self.elementscls = intg.system.elementscls
        # Get the mesh and elements
        elemap = intg.system.ele_map

        # Extract FWH Surf Geometric Data
        #fpts, qpts_dA, norm_pnorms
        intqinfo, self._int_m0 = self._get_fwhsurf_qinfo(elemap, self._int_eidxs)
        rhsqinfo, self._intrhs_m0 = self._get_fwhsurf_qinfo(elemap, self._int_rhs_eidxs)
        rcvqinfo, self._mpi_m0 = self._get_fwhsurf_qinfo(elemap, 
                                                        self._mpi_eidxs)
        sndqinfo, self._mpisnd_m0 = self._get_fwhsurf_qinfo(elemap, 
                                                        self._mpi_snd_eidxs)
        bndqinfo, self._bnd_m0 = self._get_fwhsurf_qinfo(elemap, 
                                                        self._bnd_eidxs)
        scnts, rcnts, sndqidxs = self._prepare_mpiinters_sndrcv_info(elemap)
        self._mpi_sndcnts  = scnts
        self._mpi_rcvcnts  = rcnts
        self._mpi_snd_qidxs = sndqidxs

        self._int_fplocs = intqinfo[0]
        self._intrhs_fplocs = rhsqinfo[0]
        self._mpi_fplocs = rcvqinfo[0]
        self._mpisnd_fplocs = sndqinfo[0]
        self._bnd_fplocs = bndqinfo[0]
        self.nintqpts = self._int_fplocs.shape[0]
        self.nintrhsqpts = self._intrhs_fplocs.shape[0]
        self.nmpiqpts = self._mpi_fplocs.shape[0]
        self.nmpisndqpts = self._mpisnd_fplocs.shape[0]
        self.nbndqpts = self._bnd_fplocs.shape[0]
        
        qinfo = [np.empty(0), np.empty(0), np.empty(0)]
        if self.nintqpts > 0:
            qinfo[0] = intqinfo[0]
            qinfo[1] = intqinfo[1]
            qinfo[2] = intqinfo[2]
        if self.nmpiqpts > 0:
            if np.any(qinfo[0]):
                qinfo[0] = np.vstack((qinfo[0], rcvqinfo[0])) 
                qinfo[1] = np.vstack((qinfo[1], rcvqinfo[1])) 
                qinfo[2] = np.hstack((qinfo[2], rcvqinfo[2])) 
            else:
                qinfo[0] = rcvqinfo[0]
                qinfo[1] = rcvqinfo[1]
                qinfo[2] = rcvqinfo[2]
        if self.nbndqpts > 0:
            if np.any(qinfo[0]):
                qinfo[0] = np.vstack((qinfo[0], bndqinfo[0])) 
                qinfo[1] = np.vstack((qinfo[1], bndqinfo[1])) 
                qinfo[2] = np.hstack((qinfo[2], bndqinfo[2])) 
            else:
                qinfo[0] = bndqinfo[0]
                qinfo[1] = bndqinfo[1]
                qinfo[2] = bndqinfo[2]  
        self._qpts_info = qinfo
        self.nqpts = self._qpts_info[0].shape[0]

        #prepare local srtdidxs and perms
        self._int_perm = np.empty(0)
        self._mpi_perm = np.empty(0)
        self._min_fplocs = np.empty(0)
        if self.active_fwhrank:
            idx = range(self.nqpts)
            srtdidx = fuzzysort(self._qpts_info[0].T,idx)
            #min fpts plocs
            self._min_fplocs = self._qpts_info[0][srtdidx][0]
            self._int_perm = self._compute_leftright_fptsperms(
                self._int_fplocs, self._intrhs_fplocs)

        self.gfplocs_min, self.gfplocs = self._serialise_global_fplocs(
                                        self._min_fplocs, self._qpts_info[0])

        mpi_rhs_fplocs = np.empty(0)
        if self.active_fwhedgrank:
            mpi_rhs_fplocs, _ = Alltoallv_data_arr(self.fwh_edgcomm, 
                            self.fwh_edgwrldranks, self._mpisnd_fplocs, 
                            self.ndims, self._mpi_sndcnts, self._mpi_rcvcnts,
                            self._mpi_snd_qidxs)
            if self.nmpiqpts:
                self._mpi_perm = self._compute_leftright_fptsperms(
                    self._mpi_fplocs, mpi_rhs_fplocs)

        #Debug, using analytical sources
        analytictest = self.cfg.get(self.cfgsect, 'analytic-src', None)
        self.pacoustsrc = None
        #Debug for analytical sources
        if analytictest == 'monopole':
            #shift the x,y,z coordinates so that the analytical source is inside the body
            xyz_min = np.array([1.e20]*self.ndims,'d')
            xyz_max = np.array([-1.e20]*self.ndims,'d')
            xyzshift = 0
            if np.any(self._qpts_info[0]):
                xyz_min = np.array([np.min(cr) for cr in self._qpts_info[0].T])
                xyz_max = np.array([np.max(cr) for cr in self._qpts_info[0].T])
            comm = get_comm_rank_root()[0]
            comm.Allreduce(get_mpi('in_place'),[xyz_min, MPI.DOUBLE],op=get_mpi('min'))
            comm.Allreduce(get_mpi('in_place'),[xyz_max, MPI.DOUBLE],op=get_mpi('max'))
            xyzshift = 0.5 *( xyz_min + xyz_max )
            self.observers += xyzshift

        # init the fwhsolver
        self.fwhsolver = FwhFreqDomainSolver(timeparam, self.observers, self._qpts_info, self.uinf, surftype)
        
        #init soln
        initsoln = None
        restore = False
        self._started = False
        self._last_prepared = 0.
        # init from restart
        if intg.isrestart:
            restore, initsoln = self._deserialise_root(intg, metadata)

        # Allocate/Init usolution data arrays
        self.usoln = np.empty(0)
        if self.active_fwhrank or self.active_fwhedgrank:
            self.usoln = self.fwhsolver.allocate_init_usoln_darray(restore, initsoln)
        #init step and average params
        self.samplesteps = self.fwhsolver.samplesteps
        self.stepcnt = self.fwhsolver.stepcnt
        self.avgcnt = self.fwhsolver.avgcnt

        #Debug, init the analytical source
        if analytictest == 'monopole':
            srclocs = self.cfg.getliteral(self.cfgsect, 'src-locs',[0.,0.,0.])
            srclocs += xyzshift
            srcuinf = np.array([self.uinf['rho'],self.uinf['u'][0],self.uinf['u'][1],self.uinf['u'][2],self.uinf['p']])
            srcamp = self.cfg.getfloat(self.cfgsect,'src-amp',1.)
            srcfreq = self.cfg.getfloat(self.cfgsect,'src-freq',5.)
            tperiod = 1./srcfreq
            nperiods = self.fwhsolver.ntsub * self.fwhsolver.dtsub / tperiod
            gamma = self.constvars['gamma']
            self.pacoustsrc = MonopoleSrc('smonopole',srclocs,srcuinf,srcamp,srcfreq,self.fwhsolver.ntsub,nperiods,gamma)
            if self.active_fwhrank and self.fwhrank == 0:
                co = np.sqrt(srcuinf[-1]/srcuinf[0]) 
                print(f'src locs {srclocs}')
                print(f'co {co}, rho_o {srcuinf[0]}, po {srcuinf[-1]}')
                print(f'Minf {srcuinf[1:4]/co}')
                sys.stdout.flush()
            
        self.dt_out = self.fwhsolver.dtsub # make sue to change this calling way for non-uniform fft
            # Register our output times with the integrator
            #intg.call_plugin_dt(self.dt_out)

        self.fwhwritemode = 'w' if not intg.isrestart else 'a'


    def __call__(self, intg):
        #bail out if did not reach tstart
        if intg.tcurr < self.tstart - self.tol:
            return

        #register globally that we have started the fwh computations
        self._started = True
        
        # if not an active rank return
        if (not self.active_fwhrank) and (not self.active_fwhedgrank):
            return

        self._prepare_data(intg)


    def serialise(self, intg):
        #bail out if did not reach tstart
        if not self._started:
            return {}

        if self.active_fwhrank or self.active_fwhedgrank:
            #solve fwh if needed
            self._prepare_data(intg)

        metadata = {}

        #prepare mpi communication
        comm, rank, wrldroot = get_comm_rank_root()
        fwhroot = self.fwh_wrldranks[0]
        if fwhroot != wrldroot:
            gatherroot = wrldroot
            gcomm = comm
            grank = rank
        else:
            gatherroot = fwhroot
            gcomm = self.fwh_comm
            grank = self.fwhrank
        
        sendsoln = np.empty(0)
        sendpfft = np.empty(0)
        if self.active_fwhrank :
            sendpfft= np.array([self.fwhsolver.pmagsum, self.fwhsolver.presum, self.fwhsolver.pimgsum])
            sendsoln = self.usoln[:self.fwhsolver.stepcnt].reshape(-1, self.nqpts).T
        #gather pfft of welch averaging data from all other fwh ranks to world root
        gpfftdata = Gatherv_data_arr(gcomm, grank, gatherroot, sendpfft)
        #gather soln from all other fwh ranks to world root
        gusoln = Gatherv_data_arr(gcomm, grank, gatherroot, sendsoln)
        
        #prepare timedata to be sent to wrldroot if needed
        tdata = np.empty(0)
        if rank == fwhroot:
            tdata = np.array([self._last_prepared, self.fwhsolver.ltsub, 
                                self.fwhsolver.ntsub, self.fwhsolver.stepcnt, 
                                self.fwhsolver.avgcnt, self.fwhsolver.nfreq])
        
        if fwhroot != wrldroot:
            #fwh root
            if rank == fwhroot:
                comm.Send(tdata, dest=wrldroot, tag=54)
            #world root
            if rank == wrldroot:
                tdata = np.empty(6)
                comm.Recv(tdata, source=fwhroot, tag=54)

        if rank == wrldroot:
            metadata = {
                'tlast'    : tdata[0],
                'ltsub'    : tdata[1],
                'ntsub'    : int(tdata[2]),
                'stepcnt'  : int(tdata[3]),
                'avgcnt'   : int(tdata[4]),
                'nfreq'    : int(tdata[5]),
                'region'   : self.region,
                'observers': self.observers,
                'soln'     : gusoln,
                'fptsplocs': self.gfplocs,
                'fptsplocs-min': self.gfplocs_min,
                'pfftdata' : gpfftdata
            }
        
        return metadata

    def _serialise_global_fplocs(self, minfplocs, fptsplocs):
        #prepare mpi communication
        comm, rank, root = get_comm_rank_root()
        #compute and gather min fpts-plocs from all other fwh ranks to world root
        #this is to be used for permutation of pfft data
        sendbuf = minfplocs if self.active_fwhrank else np.empty(0)
        gfplocs_min = Gatherv_data_arr(comm, rank, root, sendbuf)
        #gather flux points plocs from all other fwh ranks to world root
        sendbuf = fptsplocs if self.active_fwhrank else np.empty(0)
        gfptsplocs = Gatherv_data_arr(comm, rank, root, sendbuf)
        return gfplocs_min, gfptsplocs

    def _deserialise_root(self, intg, metadata=None):
        usoln = None
        #determine if restore is needed:
        restore = True
        if metadata['region'] != self.region:
            print(f'\nFWH region has changed in the config file,'
                    ' restore cannot be done')
            print(f'\tsaved   {metadata["region"]}')
            print(f'\tconfig  {self.region}\n')
            restore = False
        elif np.any(metadata['observers'] != self.fwhsolver.xyz_obv):
            print(f'\nFWH observers locations has changed in the config file,'
                    ' restore cannot be done')
            print(f'\tsaved   {metadata["observers"]}')
            print(f'\tconfig  {self.fwhsolver.xyz_obv}\n')
            restore = False
        elif metadata['ntsub'] != self.fwhsolver.ntsub:
            print(f'\nFWH ntsub (window nsteps) has changed in '
                        'the config file, restore cannot be done')
            print(f'\tsaved   {metadata["ntsub"]}')
            print(f'\tconfig  {self.fwhsolver.ntsub}\n') 
            restore = False
        elif metadata['ltsub'] != self.fwhsolver.ltsub:
            print(f'\nFWH ltsub (window length) has changed in '
                        'the config file, restore cannot be done\n')
            print(f'\tsaved   {metadata["ltsub"]}')
            print(f'\tconfig  {self.fwhsolver.ltsub}') 
            restore = False

        # restore saved fwh data if needed
        if restore:
            self._started = True
            #read metadata
            self._last_prepared = metadata['tlast'] 
            self.stepcnt =  nsteps = metadata['stepcnt']
            self.avgcnt = metadata['avgcnt']

            if self.active_fwhrank:
                root = 0
                #init usoln
                gsoln = metadata['soln']
                gfplocs = metadata['fptsplocs']
                #gather the current global fpts-plocs
                curr_gfplocs = Gatherv_data_arr(self.fwh_comm, self.fwhrank,
                                    root, self._qpts_info[0])

                if self.fwhrank == root:
                    nqpts_tot = int(gsoln.shape[0]/(nsteps*self.fwhnvars))
                    idx = range(nqpts_tot)
                    #sorting ids of the current fplocs
                    srtdidx = fuzzysort(curr_gfplocs.reshape(-1,self.ndims).T, idx)
                    #permutation of curr fplocs
                    perms = np.argsort(srtdidx)
                    #sorting ids of the read global fplocs
                    srtdidx = fuzzysort(gfplocs.reshape(-1,self.ndims).T, idx)
                    #permuting the global solution array to the current ordering
                    gsoln = gsoln.reshape(nqpts_tot,-1)[srtdidx][perms].reshape(-1)

                #scatter gsoln from fwh-root to all other fwh ranks
                sendbuf = None
                displ = None
                count = None
                rbufsize = int(nsteps * self.fwhnvars * self.nqpts)
                recvbuf = np.empty(rbufsize, dtype=float)
                count = self.fwh_comm.gather(rbufsize,root=0)
                if self.fwhrank == 0:
                    sendbuf = gsoln
                    count = np.array(count, dtype='i')
                    displ = np.array([sum(count[:p]) for p in range(len(count))])

                self.fwh_comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=root)
                usoln = recvbuf.reshape(self.nqpts,-1).T.reshape(nsteps,self.fwhnvars,self.nqpts)

                #init pfftdata
                #start by collecting the minimum fplocs for each fwhrank
                sendbuf = np.array(self._min_fplocs)
                curr_fplocs_min = np.empty([self.fwh_comm.size, 
                                self.ndims]) if self.fwhrank == root else None
                self.fwh_comm.Gather(sendbuf, curr_fplocs_min, root=root)

                #scattering pfftdata
                sendbuf = None
                if self.fwhrank == 0:
                    # sorting and computing permutations for current gfplocs_min
                    idx = range(self.fwh_comm.size)
                    srtdidx = fuzzysort(curr_fplocs_min.T, idx)
                    perms = np.argsort(srtdidx)
                    #computing the correct sorting of pfftdata
                    gpfftdata = metadata['pfftdata'].reshape(self.fwh_comm.size, -1)
                    gfplocs_min = metadata['fptsplocs-min']
                    idx = range(self.fwh_comm.size)
                    srtdidx = fuzzysort(gfplocs_min.reshape(-1,self.ndims).T, idx)
                    #sorting the global read pfftdata
                    sendbuf = gpfftdata[srtdidx][perms].astype(float)

                rbufsize = int(metadata['pfftdata'].size/self.fwh_comm.size)
                recvbuf = np.empty(rbufsize, dtype=float)

                self.fwh_comm.Scatter(sendbuf, recvbuf, root=root) 
                recvbuf = recvbuf.reshape(3, -1)

                metadata |= dict(pmagsum=recvbuf[0], presum=recvbuf[1], pimgsum=recvbuf[2])

            #not active fwhrank
            else:
                metadata |= dict(pmagsum=np.empty(0), presum=np.empty(0), pimgsum=np.empty(0))
            
            #init fwh solver class from restored data
            self.fwhsolver.init_from_restart(metadata)

        #no restore
        else:
            self.tstart = intg.tcurr

        return restore, usoln

    
    def _deserialise_per_rank(self, intg, metadata=None):
        usoln = None
        #determine if restore is needed:
        restore = True
        if metadata['region'] != self.region:
            print(f'\nFWH region has changed in the config file,'
                    ' restore cannot be done')
            print(f'\tsaved   {metadata["region"]}')
            print(f'\tconfig  {self.region}\n')
            restore = False
        elif np.any(metadata['observers'] != self.fwhsolver.xyz_obv):
            print(f'\nFWH observers locations has changed in the config file,'
                    ' restore cannot be done')
            print(f'\tsaved   {metadata["observers"]}')
            print(f'\tconfig  {self.fwhsolver.xyz_obv}\n')
            restore = False
        elif metadata['ntsub'] != self.fwhsolver.ntsub:
            print(f'\nFWH ntsub (window nsteps) has changed in '
                        'the config file, restore cannot be done')
            print(f'\tsaved   {metadata["ntsub"]}')
            print(f'\tconfig  {self.fwhsolver.ntsub}\n') 
            restore = False
        elif metadata['ltsub'] != self.fwhsolver.ltsub:
            print(f'\nFWH ltsub (window length) has changed in '
                        'the config file, restore cannot be done\n')
            print(f'\tsaved   {metadata["ltsub"]}')
            print(f'\tconfig  {self.fwhsolver.ltsub}') 
            restore = False

        if restore:
            self._started = True
            #read metadata
            self._last_prepared = metadata['tlast'] 
            self.stepcnt =  nsteps = metadata['stepcnt']
            self.avgcnt = metadata['avgcnt']

            if self.active_fwhrank:
                #(1) init usoln
                gsoln = metadata['soln']
                gfplocs = metadata['fptsplocs']
                #gather the current global fpts-plocs
                curr_gfplocs = Allgatherv_data_arr(self.fwh_comm, self._qpts_info[0])

                nqpts_tot = int(gsoln.shape[0]/(nsteps*self.fwhnvars))
                idx = range(nqpts_tot)
                #sorting ids of the current fplocs
                srtdidx = fuzzysort(curr_gfplocs.reshape(-1,self.ndims).T, idx)
                #permutation of curr fplocs
                perms   = np.argsort(srtdidx)
                #sorting ids of the read global fplocs
                srtdidx = fuzzysort(gfplocs.reshape(-1,self.ndims).T, idx)
                #permuting the global solution array to the current ordering
                gsoln   = gsoln.reshape(nqpts_tot,-1)[srtdidx][perms]

                #pick relevant rank usoln from the global gsoln array             
                count   = self.fwh_comm.allgather(self.nqpts)
                count   = np.array(count, dtype='i')
                iqstart = sum(count[:self.fwhrank]) 
                iqend   = iqstart + count[self.fwhrank]
                usoln   = gsoln[iqstart: iqend].T.reshape(-1, self.fwhnvars, self.nqpts)

                #(2) init pfftdata
                #start by collecting the minimum fplocs for each fwhrank
                sendbuf = np.array(self._min_fplocs)
                curr_fplocs_min = np.empty([self.fwh_comm.size, self.ndims])
                self.fwh_comm.Allgather(sendbuf, curr_fplocs_min)

                # sorting and computing permutations for current gfplocs_min
                idx = range(self.fwh_comm.size)
                srtdidx = fuzzysort(curr_fplocs_min.T, idx)
                perms = np.argsort(srtdidx)
                #computing the correct sorting of pfftdata
                gpfftdata = metadata['pfftdata'].reshape(self.fwh_comm.size, -1)
                gfplocs_min = metadata['fptsplocs-min']
                idx = range(self.fwh_comm.size)
                srtdidx = fuzzysort(gfplocs_min.reshape(-1,self.ndims).T, idx)
                #sorting the global read pfftdata
                srtd_pfftdata = gpfftdata[srtdidx][perms]
                #picking the relevant part from the pfftdata per rank
                recvbuf = srtd_pfftdata[self.fwhrank].reshape(3, -1)
                
                metadata |= dict(pmagsum=recvbuf[0], presum=recvbuf[1], pimgsum=recvbuf[2])
            
            #not active fwhrank
            else:
                metadata |= dict(pmagsum=np.empty(0), presum=np.empty(0), pimgsum=np.empty(0))
            
            #init fwh solver class from restored data
            self.fwhsolver.init_from_restart(metadata)

        #not restore
        else:
            self.tstart = intg.tcurr

        return restore, usoln


    def _prepare_data(self,intg):
        # Already done for this step
        if self._last_prepared >= intg.tcurr:
            return

        # If sampling is not due, return
        dosample = intg.nacptsteps % self.samplesteps == 0 
        if not dosample:
            return
        
        #current time according to fwh
        tcurrfwh = np.abs(intg.tcurr - self.tstart)
        #total time for completed windows
        ttotwind = (self.avgcnt - 1)*self.fwhsolver.shift*self.fwhsolver.ltsub
        #local time within one window
        self.tlocwind = np.abs(tcurrfwh - ttotwind) 
        step = self.fwhsolver.stepcnt
        intqf = self.nintqpts
        mpiq0, mpiqf = intqf, intqf + self.nmpiqpts
        bndq0, bndqf = mpiqf, mpiqf + self.nbndqpts
        intusoln = self.usoln[step, ...,      : intqf]
        mpiusoln = self.usoln[step, ..., mpiq0: mpiqf]
        bndusoln = self.usoln[step, ..., bndq0: bndqf]
        int_rhsusoln = np.empty((self.fwhnvars, self.nintrhsqpts))
        mpi_sndusoln = np.empty((self.fwhnvars, self.nmpisndqpts))
        
        if self.active_fwhedgrank:
            #mpi interfaces with outside elements (rhs)
            if self.nmpisndqpts:
                if self.pacoustsrc:
                    self.pacoustsrc.update_usoln_onestep(self._mpisnd_fplocs, tcurrfwh, mpi_sndusoln)
                else:
                    self._update_usoln_onestep(intg, self._mpisnd_m0, self._mpi_snd_eidxs, mpi_sndusoln)
            #send/recv in an alltoall for averaging over mpi interfaces
            mpi_rhsusoln, allreq = Alltoallv_data_arr(self.fwh_edgcomm, self.fwh_edgwrldranks, 
                                                   mpi_sndusoln.T, self.fwhnvars, self._mpi_sndcnts, 
                                                   self._mpi_rcvcnts, self._mpi_snd_qidxs, 'nonblocking')
            mpi_rhsusoln = mpi_rhsusoln.T

            #mpi interfaces with inside elements (lhs)
            if self.nmpiqpts:
                if self.pacoustsrc:
                    self.pacoustsrc.update_usoln_onestep(self._mpi_fplocs, tcurrfwh, mpiusoln)
                else:
                    self._update_usoln_onestep(intg, self._mpi_m0, self._mpi_eidxs, mpiusoln)

        if self.pacoustsrc:
            #interior interfaces qpts
            if self.nintqpts:
                #inside elements (lhs)
                self.pacoustsrc.update_usoln_onestep(self._int_fplocs, tcurrfwh, intusoln)
                #outside elements (rhs)
                self.pacoustsrc.update_usoln_onestep(self._intrhs_fplocs, tcurrfwh, int_rhsusoln)
            #boundary interfaces
            if self.nbndqpts:
                self.pacoustsrc.update_usoln_onestep(self._bnd_fplocs, tcurrfwh, bndusoln)
        else:
            #interior interfaces qpts
            if self.nintqpts:
                #inside elements (lhs)
                self._update_usoln_onestep(intg, self._int_m0, self._int_eidxs, intusoln)
                #outside elements (rhs)
                self._update_usoln_onestep(intg, self._intrhs_m0, self._int_rhs_eidxs, int_rhsusoln)
            #boundary interfaces
            if self.nbndqpts:
                self._update_usoln_onestep(intg, self._bnd_m0, self._bnd_eidxs, bndusoln)
                #bndusoln = #apply boundary condition
        
        #averaging of solution over interfaces
        if self.nintqpts:
            self._usoln_inters_average(self._int_perm, int_rhsusoln, intusoln)

        if self.active_fwhedgrank:  
            allreq.wait()
            if self.nmpiqpts:
                self._usoln_inters_average(self._mpi_perm, mpi_rhsusoln, mpiusoln)
            
        self.fwhsolver.sampled()
        
        #Debug
        if self.active_fwhrank and self.fwhrank == 0:
            print(f'FWH sampled, wstep {self.fwhsolver.stepcnt-1}, wtime {np.round(self.tlocwind,5)}, tcurrfwh {np.round(tcurrfwh,5)}, currtime {np.round(intg.tcurr,5)}', flush=True)
            
        # compute fwh solution if due
        docompute = self.fwhsolver.ltsub - self.fwhsolver.dtsub - self.tlocwind <= self.tol 
        if docompute:
            if self.active_fwhrank:
                self.fwhsolver.compute_fwh_solution(self.usoln)
                nwindows = self.fwhsolver.avgcnt - 1 
                pfft = self.fwhsolver.pfft
                if self.fwhrank != 0:
                    self.fwh_comm.Reduce(pfft, None, op=get_mpi('sum'), root=0)
                else:
                    self.fwh_comm.Reduce(get_mpi('in_place'), pfft, op=get_mpi('sum'), root=0)
                                                                            
                #compute spectrums
                if self.fwhrank == 0:
                    amp = np.abs(pfft)
                    df = self.fwhsolver.freq[1]
                    # computing power spectrum outputs
                    psd = compute_psd(amp,df)
                    spl, oaspl = compute_spl(self.pref,amp)
                    print(f'\nFWH computed (Naver {nwindows-1}), ...........................')

                    # Writing spectral results
                    bname = os.path.join(self.basedir,f'{self.basename}'+'_ob{ob}.csv')
                    nwindarr = np.tile(str(nwindows),len(pfft[0]))
                    for ob in range(self.nobserv):
                        wdata = np.array([self.fwhsolver.freq,np.abs(pfft[ob,:]),np.angle(pfft[ob,:]),psd[ob,:],spl[ob,:],nwindarr]).T
                        fname = bname.format(ob=ob)
                        header = ','.join(['#Frequency (Hz)', ' Magnitude (Pa)', ' Phase (rad)'])
                        header += ', PSD (Pa^2/Hz)' if self.fwhsolver.psd_scale_mode == 'density' else ' POWER-SPECTRUM (Pa^2/Hz)'
                        header += f', SPL (dB), Nwindow'
                        write_fftdata(fname,wdata,header=header,mode=self.fwhwritemode)

                    #Debug, exact source solution
                    if self.pacoustsrc:
                        freq_ex, pfft_ex = self.pacoustsrc.exact_solution(self.observers)
                        bname = os.path.join(self.basedir,f'{self.basename}'+'exact_ob{ob}.csv')
                        for ob in range(self.nobserv):
                            wdata = np.array([freq_ex,np.abs(pfft_ex[ob,:]),np.angle(pfft_ex[ob,:]),nwindarr]).T
                            fname = bname.format(ob=ob)
                            header = ','.join(['#Frequency (Hz)', ' Magnitude (Pa)', ' Phase (rad)', ' Nwindow'])
                            write_fftdata(fname,wdata,header=header,mode=self.fwhwritemode)
                    print(f'FWH written .......................................\n')
                    self.fwhwritemode ='a'
            elif self.active_fwhedgrank:
                self.fwhsolver.update_after_onewindow()
            self.stepcnt = self.fwhsolver.stepcnt
            self.nwindows = self.fwhsolver.avgcnt - 1 
            self.avgcnt = self.nwindows + 1
    
        self._last_prepared = intg.tcurr
        self._started = True


    def _extract_drum_region_eset(self, intg, drum):
        elespts = {}
        eset = {}
        
        drum = np.asarray(drum)
        nstations = drum.shape[0]-1

        mesh = intg.system.mesh
        for etype in intg.system.ele_types:
            elespts[etype] = mesh[f'spt_{etype}_p{intg.rallocs.prank}']

        slope = []
        for i in range(0, nstations):
            slope.append((drum[i+1, 1] 
                            - drum[i, 1])/(drum[i+1, 0] - drum[i, 0]))
        
        # shift the z origin to have a symmetric drum around the origin
        pz_max = -1e20
        pz_min = 1e20
        pz_shift=0.
        if self.ndims==3:
            for etype in intg.system.ele_types:
                pts = np.moveaxis(elespts[etype], 2, 0)
                for px,py,pz in zip(pts[0],pts[1],pts[2]):
                    pz_max = np.max([pz_max, np.max(pz)])
                    pz_min = np.min([pz_min, np.min(pz)])

            #if pz_min > 1.e-16: 
            pz_shift = 0.5*(pz_min + pz_max)

        #collect the region full eset
        for etype in intg.system.ele_types:
            pts = np.moveaxis(elespts[etype], 2, 0)
            
            # Determine which points are inside the fwh surface
            inside = np.zeros(pts.shape[1:], dtype=np.bool)

            if self.ndims == 3:
                zippts = zip(pts[0], pts[1], pts[2])
            else:
                zippts = zip(pts[0], pts[1], np.zeros(pts[0].shape))

            for px, py, pz in zippts:
                pz -= pz_shift
                rmesh = np.sqrt(py*py + pz*pz)
                for i in range(0, nstations):
                    xcond = (drum[i, 0] <= px) & (px <= drum[i+1, 0])
                    rdrum = (drum[i, 1] + (px - drum[i, 0])*slope[i])
                    inside += xcond & (rmesh < rdrum)

            if np.sum(inside):
                eset[etype] = np.any(inside, axis=0).nonzero()[0]

        return eset

    def _extract_bound_region_eset(self, intg, bcname):
        comm, rank, root = get_comm_rank_root()

        # Get the mesh and prepare the element set dict
        mesh = intg.system.mesh
        eset = defaultdict(list)

        # Boundary of interest
        bc = f'bcon_{bcname}_p{intg.rallocs.prank}'

        # Ensure the boundary exists
        bcranks = comm.gather(bc in mesh, root=root)
        if rank == root and not any(bcranks):
            raise ValueError(f'Boundary {bcname} does not exist')

        if bc in mesh:
            # Determine which of our elements are on the boundary
            for etype, eidx in mesh[bc][['f0', 'f1']].astype('U4,i4'):
                eset[etype].append(eidx)
        return eset

    def _write_fwh_surface_geo(self, intg, eset, inters_eidxs):
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
        vtuwriter = VTUSurfWriter(intg, inters_eidxs)
        # Write VTU file:
        vtumfname = os.path.splitext(self.fwhmeshfname)[0]
        vtuwriter.write(vtumfname)

    def _prepare_surfmesh(self, intg, eset, surftype='permeable'):
        self._inside_eset  = defaultdict(list)
        self._eidxs        = defaultdict(list)  
        self._int_eidxs    = defaultdict(list)
        self._int_rhs_eidxs = defaultdict(list)
        self._mpi_eidxs = defaultdict(list)  
        self._mpi_snd_eidxs = defaultdict(list)
        self._bnd_eidxs    = defaultdict(list) 
        mesh = intg.system.mesh
        # Collect fwh interfaces and their info
        if surftype == 'permeable':
            self._collect_intinters(intg.rallocs, mesh, eset)
            self._collect_mpiinters(intg.rallocs, mesh, eset)
        self._collect_bndinters(intg.rallocs, mesh, eset)

        self._eidxs = {k: np.array(v) for k, v in self._eidxs.items()}
        self._int_eidxs = {k: np.array(v) for k, v in self._int_eidxs.items()}
        self._int_rhs_eidxs = {
            k: np.array(v) for k, v in self._int_rhs_eidxs.items()}
        self._mpi_snd_eidxs = {
            k: np.array(v) for k, v in self._mpi_snd_eidxs.items()}
        self._mpi_eidxs = {
            k: np.array(v) for k, v in self._mpi_eidxs.items()}
        self._bnd_eidxs = {k: np.array(v) for k, v in self._bnd_eidxs.items()} 


    def _collect_intinters(self, rallocs, mesh, eset):
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
                    self._int_eidxs[etype, fidx].append(eidx)
                    self._int_rhs_eidxs[ifaceR[0],ifaceR[2]].append(ifaceR[1])
                else:
                    etype, eidx, fidx = ifaceR[0:3]
                    self._int_eidxs[etype, fidx].append(eidx)
                    self._int_rhs_eidxs[ifaceL[0],ifaceL[2]].append(ifaceL[1])

                self._eidxs[etype, fidx].append(eidx)
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)

            # periodic faces:
            elif (flagL & flagR) & (ifaceL[3] != 0 ):   
                etype, eidx, fidx = ifaceL[0:3]
                self._int_eidxs[etype, fidx].append(eidx)
                self._eidxs[etype, fidx].append(eidx)
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)
                #add both right and left faces for vtu writing
                etype, eidx, fidx = ifaceR[0:3]
                self._int_eidxs[etype, fidx].append(eidx)
                self._eidxs[etype, fidx].append(eidx)
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)
                self._int_rhs_eidxs[ifaceL[0],ifaceL[2]].append(ifaceL[1])
                self._int_rhs_eidxs[ifaceR[0],ifaceR[2]].append(ifaceR[1])

            
    def _collect_mpiinters(self, rallocs, mesh, eset):
        comm = get_comm_rank_root()[0]
        prank = rallocs.prank
        # send flags
        for rhs_prank in rallocs.prankconn[prank]:
            conkey = f'con_p{prank}p{rhs_prank}'
            mpiint = mesh[conkey].astype('U4,i4,i1,i2').tolist()
            flagL = np.zeros((len(mpiint)), dtype=bool)
            for findex, ifaceL in enumerate(mpiint):
                etype, eidx, fidx = ifaceL[0:3]
                flagL[findex] = (eidx in 
                                    eset[etype]) if etype in eset else False
            rhs_mrank = rallocs.pmrankmap[rhs_prank]
            comm.Send(flagL, rhs_mrank, tag=52)

        # receive flags and collect mpi interfaces
        self._mpi_rcvrankmap = defaultdict(list)
        self._mpi_sndrankmap = defaultdict(list)

        #loop over the mpi interface ranks
        for rhs_prank in rallocs.prankconn[prank]:
            conkey = f'con_p{prank}p{rhs_prank}'
            mpiint = mesh[conkey].astype('U4,i4,i1,i2').tolist()
            flagR = np.empty((len(mpiint)), dtype=bool)
            rhs_mrank = rallocs.pmrankmap[rhs_prank] 
            comm.Recv(flagR, rhs_mrank, tag=52)

            #loop over the rank connectivity
            for findex, ifaceL in enumerate(mpiint):

                etype, eidx, fidx = ifaceL[0:3]
                flagL = (eidx in eset[etype]) if etype in eset else False
                # add info if it is an fwh edge, i.e., 
                # having either flagL or flagR or 
                # both(periodic case) to be True
                if flagL and not flagR[findex] :
                    self._eidxs[etype, fidx].append(eidx)
                    if eidx not in  self._inside_eset[etype]:
                        self._inside_eset[etype].append(eidx)
                    self._mpi_eidxs[etype, fidx].append(eidx)
                    self._mpi_rcvrankmap[etype, fidx].append(rhs_mrank)

                #an interface to be sent to rhs owning rank
                elif flagR[findex] and not flagL:
                    self._mpi_snd_eidxs[etype, fidx].append(eidx)
                    self._mpi_sndrankmap[etype, fidx].append(rhs_mrank)

                #periodic mpi interfaces
                elif (flagL and flagR[findex]) and ifaceL[-1] !=0 :
                    self._eidxs[etype, fidx].append(eidx)
                    if eidx not in  self._inside_eset[etype]:
                        self._inside_eset[etype].append(eidx)
                    
                    self._mpi_eidxs[etype, fidx].append(eidx)
                    self._mpi_snd_eidxs[etype, fidx].append(eidx)
                    self._mpi_rcvrankmap[etype, fidx].append(rhs_mrank)
                    self._mpi_sndrankmap[etype, fidx].append(rhs_mrank)
        
    def _collect_bndinters(self, rallocs, mesh, eset):
        prank = rallocs.prank
        for f in mesh:
            if (m := re.match(f'bcon_(.+?)_p{prank}$', f)):
                bname = m.group(1)
                bclhs = mesh[f].astype('U4,i4,i1,i2').tolist()
                for ifaceL in bclhs:
                    etype, eidx, fidx = ifaceL[0:3]
                    flagL = (eidx in eset[etype]) if etype in eset else False
                    if flagL:
                        self._bnd_eidxs[etype, fidx].append(eidx)
                        self._eidxs[etype, fidx].append(eidx)
                        if eidx not in  self._inside_eset[etype]:
                            self._inside_eset[etype].append(eidx)


    def _build_fwh_mpi_comms(self, eidxs, mpieidxs, mpi_sndeidxs):
        # fwhranks own all fwh edges and hence are 
        # the ones that perform acoustic solve
        # fwhedgeranks are ranks who touches an fwh edge 
        # but not necessarily owns an edge in general.
        # However, they can be in both fwhedgeranks and
        # fwhranks list if they happen to be both touching 
        # an edge as an outsider and has inside cells (insiders) 
        # and hence own some edges.
        self.fwh_wrldranks = []
        self.fwh_edgwrldranks = []
        comm, rank, _ = get_comm_rank_root()

        # Determine active ranks for fwh computations
        self.active_fwhrank = True if eidxs else False
        rank_is_active_list = comm.allgather(self.active_fwhrank)
        for i, flag in enumerate(rank_is_active_list):
            if flag:
                self.fwh_wrldranks.append(i)

        # Determine fwh mpi edge/interface sharing ranks
        self.active_fwhedgrank = True if mpieidxs or mpi_sndeidxs else False
        rank_is_active_list = comm.allgather(self.active_fwhedgrank)
        for i, flag in enumerate(rank_is_active_list):
            if flag:
                self.fwh_edgwrldranks.append(i) 
        
        # Constructing sub-communicators and sub-groups
        fwh_group = comm.group.Incl(self.fwh_wrldranks)
        fwh_edgegroup = comm.group.Incl(self.fwh_edgwrldranks)
        fwh_comm = comm.Create(fwh_group)
        fwh_edgecomm = comm.Create(fwh_edgegroup)
        if rank not in self.fwh_wrldranks:
            fwh_group = MPI.GROUP_NULL
            fwh_comm = MPI.COMM_NULL
        if rank not in self.fwh_edgwrldranks:
            fwh_edgegroup = MPI.GROUP_NULL
            fwh_edgecomm = MPI.COMM_NULL

        if fwh_comm is not MPI.COMM_NULL:
            self.fwhrank = fwh_comm.rank
        if fwh_edgecomm is not MPI.COMM_NULL:
            self.fwhedgrank = fwh_edgecomm.rank
        
        if self.active_fwhrank and self.fwhrank == 0:
            print(f'\n{len(self.fwh_wrldranks)} '
                f'fwh surface mranks: {self.fwh_wrldranks}')
            print(f'{len(self.fwh_edgwrldranks)}'
                f' fwh mpi/edge ranks: {self.fwh_edgwrldranks}\n')
        sys.stdout.flush()

        return fwh_comm, fwh_edgecomm

    def _prepare_mpiinters_sndrcv_info(self, elemap):
        #prepare send/recv permutations
        sndqidxs= defaultdict(list)
        rcvqidxs = defaultdict(list)
        iq = 0
        for (etype, fidx), eidlist in self._mpi_eidxs.items():
            nfacefpts = elemap[etype].basis.nfacefpts[fidx]
            for id, eid in enumerate(eidlist):
                rcvrank = self._mpi_rcvrankmap[etype, fidx][id]
                for i in range(nfacefpts):
                    rcvqidxs[rcvrank].append(iq)
                    iq += 1 
        iq = 0
        for (etype, fidx), eidlist in self._mpi_snd_eidxs.items():
            nfacefpts = elemap[etype].basis.nfacefpts[fidx]
            for id, eid in enumerate(eidlist):
                sndrank = self._mpi_sndrankmap[etype, fidx][id]
                for i in range(nfacefpts):
                    sndqidxs[sndrank].append(iq)
                    iq += 1

        #prepare send/recv counts for mpi edge interfaces
        nedgranks = len(self.fwh_edgwrldranks)
        sndcnts = [0]*nedgranks
        rcvcnts = [0]*nedgranks
        for ir, rk in enumerate(self.fwh_edgwrldranks):
            sndcnts[ir] = len(sndqidxs[rk]) if rk in sndqidxs else 0
            rcvcnts[ir] = len(rcvqidxs[rk]) if rk in rcvqidxs else 0

        return sndcnts, rcvcnts, sndqidxs

    def _get_fwhsurf_qinfo(self, elemap, eidxs):
        m0 = {}
        ndims = self.ndims
        nqpts = 0
        nfaces = 0

        for (etype, fidx), eidlist in eidxs.items():
            eles = elemap[etype]
            
            if (etype, fidx) not in m0:
                facefpts = eles.basis.facefpts[fidx]
                m0[etype, fidx] = eles.basis.m0[facefpts]
                qwts = eles.basis.fpts_wts[facefpts]
                nfacefpts = eles.basis.nfacefpts[fidx]

            nfaces_ = len(eidlist)
            mpn = np.empty((nfaces_, nfacefpts))
            npn = np.empty((nfaces_, nfacefpts, ndims))
            fplocs = np.empty((nfaces_, nfacefpts, ndims))
            
            for ie, eidx in enumerate(eidlist):
                # save flux pts coordinates:
                fplocs[ie] = eles.plocfpts[facefpts, eidx] 
                # Unit physical normals and their magnitudes (including |J|)
                mpn[ie] = eles.get_mag_pnorms(eidx, fidx)
                npn[ie] = eles.get_norm_pnorms(eidx, fidx)
            
            qdA = np.einsum('i,ji->ji', qwts, mpn)
            
            if nqpts == 0:
                dA = qdA.reshape(-1) 
                fpts_plocs  = fplocs.reshape(-1, ndims)  
                norm_pnorms = npn.reshape(-1, ndims) 
            else:
                dA = np.hstack((dA, qdA.reshape(-1)))
                fpts_plocs  = np.vstack((fpts_plocs, 
                                            fplocs.reshape(-1, ndims)))
                norm_pnorms = np.vstack((norm_pnorms, 
                                            npn.reshape(-1, ndims)))

            nqpts  += (nfacefpts*nfaces_)
            nfaces += nfaces_

        qinfo = [fpts_plocs, norm_pnorms, dA] if m0 else [np.empty(0)]*3
        
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

        return qinfo, m0   

    def _compute_leftright_fptsperms(self, lfplocs, rfplocs):
        idx = range(lfplocs.shape[0])
        srtdidx = fuzzysort(lfplocs.T, idx) if idx else []
        perm = np.argsort(srtdidx)
        idx = range(rfplocs.shape[0])
        srtdidx = fuzzysort(rfplocs.T, idx) if idx else []
        convperm = np.array(srtdidx)[perm]
        return convperm

    def _update_usoln_onestep(self, intg, m0_dict, eidxs, usoln):
        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))
        ndims, nvars = self.ndims, self.nvars

        fIo = 0
        for (etype, fidx), m0 in m0_dict.items(): 
            # Get the interpolation operator
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][...,eidxs[etype, fidx]]

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1) # nvars, nfpts, nfaces

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
        
    def _usoln_inters_average(self, rhsperm, rsoln, lsoln):
        lsoln += rsoln.T[rhsperm].T
        lsoln *= 0.5
        return lsoln
        

class FwhSolverBase(object):
    #if more windows are needed, they can be customally added. 
    # A larger list of windows is available in scipy
    windows = {
        # rectangle window
        'None': (lambda s: np.ones(s), {'density': 1., 'spectrum': 1.}),
        'hanning': (lambda s: np.hanning(s),   
                    {'density': np.sqrt(8./3.), 'spectrum': 2.}),
        'hamming': (lambda s: np.hamming(s),   
                   {'density': 50.*np.sqrt(3974.)/1987., 'spectrum': 50./27.}),
        'blackman': (lambda s: np.blackman(s),  
                   {'density': 50.*np.sqrt(3046.)/1523., 'spectrum': 50./21.}),
        'bartlett': (lambda s: np.bartlett(s),  
                    {'density': np.sqrt(3.), 'spectrum': 2.}),
        }
    tol = 1e-12
    pref = 2e-5

    def __init__(self,timeparam,observers,surfdata,Uinf,surftype='permeable'):
        
        self.nobserv = len(observers) # number of observers
        self.xyz_obv = observers
        self.uinf = Uinf
        self.surfdata = surfdata
        self.xyz_src, self.qnorms, self.qdA = self.surfdata
        self.nvars = len(self.uinf['u'])+2
        self.nqpts = np.asarray(self.surfdata[0]).shape[0]
        self.ndims = len(observers[0])
        self.solidsurf = True if surftype == 'solid' else False
        
        if (timeparam.ltsub + self.tol) <= timeparam.dtsim:
            raise ValueError(f'ltsub window length {timeparam.ltsub}'
                        f' is too short or less than simulation time step'
                        f' {timeparam.dtsim}')
        if timeparam.shift > 1.:
            raise  ValueError(f'window overlap/shift {timeparam.shift} cannot '
                                'exceed one, please adjust it as necessary')
        if not timeparam.window in self.windows:
            raise ValueError(f'{timeparam.window} window type is not'
                    ' implemented, please choose ''{None, hanning, hamming,'
                    ' blackman, bartlett}''')

        self._prepare_fft_param(timeparam)

    def _prepare_fft_param(self,timeparam):
        ltsub = timeparam.ltsub
        dtsub = timeparam.dtsub 
        self.shift = shift = timeparam.shift
        if timeparam.window in list(self.windows)[1:]:
            self.window = window = timeparam.window
        else:
            self.window = window = None
        self.psd_scale_mode = scaling_mode = timeparam.psd_scale_mode
        self.dtsim = dtsim = timeparam.dtsim
        
        # Adjust inputs
        ntsub = ltsub/dtsub
        #(1) dtsub
        self.dtsub = dtsub = int(np.rint(dtsub/dtsim))*dtsim  if dtsub > dtsim else dtsim
        #(2) compute ntsub as a power of 2 number
        N = int(np.rint(ltsub/dtsub))
        self.ntsub = ntsub = self.get_num_nearest_pow_of_two(N)
        #self.ntsub = ntsub = N
        #(3): adjust the window length
        self.ltsub = ltsub = ntsub * dtsub

        #(4) Adjusting shift parameters for window averaging and overlapping
        # partial (0.01 < shift < 1) or complete overlap (shift = 1)
        if shift > 0.01:
            self.avgcnt = 1
            self.averaging = True
            self.ntoverlap = int(np.rint(ntsub*shift))
        # no averaging or shifting, shift < 0.01 
        else:
            self.avgcnt = 1
            self.averaging= False
            self.ntoverlap = ntsub    
        self.stepcnt = 0
        self.samplesteps = int(self.dtsub/self.dtsim)
        
        #(5) window function params
        #since we are using windows for spectral analysis we do not use the last data entry
        self.wwind = self.windows[window][0](ntsub+1)[:-1] if window else window
        self.windscale = self.windows[window][1][scaling_mode] if window else window

    def print_spectraldata(self):
        print(f'\n--------------------------------------')
        print(f'       Adjusted FFT parameters ')
        print(f'--------------------------------------')
        print(f'sample steps: {self.samplesteps}')
        print(f'sample freq : {1./self.dtsub} Hz')
        print(f'delta freq  : {1./self.ltsub} Hz')
        print(f'dt window   : {self.dtsub} sec')
        print(f'Lt window   : {self.ltsub} sec')
        print(f'Nt window   : {self.ntsub}')
        print(f'Nt shifted  : {self.ntoverlap}')
        if self.averaging:
            print(f'PSD Averaging is \'activated\'')
            print(f'Naver  : {self.avgcnt}')
        else:
            print(f'PSD Averaging is \'not activated\'')
        print(f'window function is \'{self.window}\'')
        print(f'wwind {self.wwind}, wscale {self.windscale}')
        print(f'psd scaling mode is \'{self.psd_scale_mode}\'\n')
        return

    def allocate_init_usoln_darray(self, restore=False, indata=None):
        usoln = np.empty((self.ntsub,self.nvars,self.nqpts))
        if restore:
            usoln[:self.stepcnt] = indata

        if get_comm_rank_root()[1] == 0:
            self.print_spectraldata()

        return usoln

    def check_sample(self,nstep):
        return nstep % self.samplesteps == 0

    def sampled(self, nstep=1):
        self.stepcnt += nstep
    
    def check_compute(self):
        return self.stepcnt == self.ntsub

    def update_after_onewindow(self):
        #update the step counter
        self.stepcnt = self.ntsub - self.ntoverlap
        if self.averaging:
            self.avgcnt += 1


    #-FFT utilities
    @staticmethod
    def rfft(udata_):
        dsize = np.size(udata_)
        # freqsize: size of half the spectra with positive frequencies
        freqsize = int(dsize/2) if dsize%2 == 0 else int(dsize-1)/2
        ufft = np.fft.rfft(udata_)/freqsize 
        return ufft

    @staticmethod
    def welch_accum(ufft, magsum, realsum, imgsum, mode='density'):
        mag = np.abs(ufft)*np.abs(ufft) if mode == 'density' else np.abs(ufft)
        magsum   += mag
        realsum  += np.real(ufft)
        imgsum   += np.imag(ufft)
        return magsum, realsum, imgsum

    @staticmethod
    def compute_welch_average(nwindows, umag, ureal, uimg, mode='density'):
        mag = np.sqrt(umag / nwindows) if mode=='density' else umag/nwindows
        phase = np.arctan2( uimg, ureal)
        ufft = mag * np.exp(1j * phase)
        return ufft

    @staticmethod
    def get_num_nearest_pow_of_two(N):
        exponent = int(np.log2(N))
        diff0 = np.abs(N-pow(2,exponent))
        diff1 = np.abs(N-pow(2,exponent+1))
        return pow(2,exponent) if diff0 < diff1 else pow(2,exponent+1)

    @staticmethod
    def _surface_integrate(qdA,p):
        #i=nobservers, j=nfreq or ntime, k=nqpts
        return np.einsum('ijk,k->ij',p,qdA)

    def compute_fwh_solution(self, *args, **kwds):
        pass


class FwhFreqDomainSolver(FwhSolverBase):

    def __init__(self, timeparm, observers, surfdata, Uinf, 
                 surftype='permeable'):
        super().__init__(timeparm, observers, surfdata, Uinf, surftype)
        
        # compute distance vars for fwh
        mR, mRs, nR, nRs = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        if np.any(self.xyz_src):
            Mi = self.uinf['Mach']
            srclocs = self.xyz_src
            oblocs = self.xyz_obv
            mR, mRs, nR, nRs = self.compute_distance_vars(srclocs, oblocs, Mi)
        self.magR, self.magRs, self.nRvec, self.nRsvec = mR, mRs, nR, nRs
        # compute frequency parameters
        self.freq  = np.fft.rfftfreq(self.ntsub, self.dtsub)
        self.omega = 2*np.pi*self.freq
        self.kwv = self.omega / self.uinf['c']
        self.nfreq = np.size(self.freq)

        # Init fwh outputs 
        self.pfft = np.empty((self.nobserv, self.nfreq), dtype=np.complex128)
        self.pmagsum = np.zeros((self.nobserv, self.nfreq))
        self.presum = np.zeros((self.nobserv, self.nfreq))
        self.pimgsum = np.zeros((self.nobserv, self.nfreq))
    
    def init_from_restart(self, initdata):
        self.stepcnt = initdata['stepcnt']
        self.avgcnt  = initdata['avgcnt']
        self.ntsub   = initdata['ntsub']
        self.pmagsum = initdata['pmagsum'].reshape(self.nobserv, -1)
        self.presum  = initdata['presum'].reshape(self.nobserv, -1)
        self.pimgsum = initdata['pimgsum'].reshape(self.nobserv, -1)

        return


    @staticmethod
    def compute_distance_vars(xyz_src, xyz_ob, Minf):
        nobserv = xyz_ob.shape[0]
        ndims = xyz_ob[0].shape[0]
        magMinf = np.sqrt(sum(m*m for m in Minf)) # |M|
        # inverse of Prandtl Glauert parameter
        gm_ = 1./np.sqrt(1.0 - magMinf*magMinf) 
        gm2_ = gm_*gm_
        # dr = x_observer - x_source
        dr = np.array([ob - xyz_src for ob in xyz_ob]).reshape(-1, ndims)    
        magdr = np.sqrt(sum((dr*dr).T)) # distance |dr|
        # normalized unit dr
        ndr = np.array([idr/ir for idr,ir in zip(dr,magdr)]) 
        Minf_ndr = sum((Minf*ndr).T) # Minfty.r_hat
        Rs = (magdr/gm_)*np.sqrt(1. + gm2_*Minf_ndr*Minf_ndr)   # |R*|
        R = gm2_*(Rs - magdr*Minf_ndr)               # |R|
        Minf2_ndr = np.einsum('i,j->ji', Minf, Minf_ndr)
        rrs = magdr/(gm2_*Rs)
        mr = ndr + Minf2_ndr*gm2_
        nRs = np.einsum('i,ij->ij', rrs, mr) # normalized unit R*
        nR = gm2_ * (nRs-Minf)    # normalized unit R
        if not (R.shape[0] == nobserv):
            nR = nR.reshape(nobserv,-1, ndims)
            nRs = nRs.reshape(nobserv,-1, ndims)
            R = R.reshape(nobserv, -1)
            Rs = Rs.reshape(nobserv, -1)
 
        return R,Rs,nR,nRs

    def compute_fwh_solution(self, usoln):
        #compute fluxes in time domain
        Q, F = self._compute_surface_fluxes(usoln)
        #compute fwh pressure solution in frequency domain
        self.pfft = self._compute_observer_pfft(Q,F)
        # use welch method if averaging
        if self.averaging:
            self.welch_accum(self.pfft, self.pmagsum, self.presum, 
                            self.pimgsum, self.psd_scale_mode)
            if self.avgcnt > 1:
                self.pfft = self.compute_welch_average(self.avgcnt, 
                self.pmagsum, self.presum, self.pimgsum, self.psd_scale_mode)
            #update the averaging counter
            self.avgcnt += 1
        #update the step counter
        self.stepcnt = self.ntsub - self.ntoverlap

    def _compute_surface_fluxes(self, usoln):
        ndims =  self.ndims
        # define local vars
        qnorms = self.qnorms
        cinf = self.uinf['c']
        pinf = self.uinf['p']
        rhoinf = self.uinf['rho']
        uinf = np.array(self.uinf['u'])
        # compute soln vars
        u = np.swapaxes(usoln[:, 1: ndims+1,:], 1, 2) - uinf #u' perturbed vel
        p = usoln[:, -1, :] - pinf # p' fluctuation pressure
        # density, another formulation will just use rho in usoln[:,0,:]
        rho_tot = rhoinf + p/(cinf*cinf) 
        # compute normal velocities
        un  = np.einsum('ij,kij->ki', qnorms, u) # normal flow velocity
        Uin = sum((uinf*qnorms).T)  # normal Uinfty
        vn = un + Uin if self.solidsurf else 0.
        dvn = vn - Uin  # relative surface velocity
        dUn = un - dvn  # relative normal flow velocity
        # compute temporal fluxes
        rdun = rho_tot*dUn
        umuinf = u - uinf
        rhouinf = rhoinf*uinf
        Q = rdun + rhoinf*dvn
        F = np.einsum('ij,ijk->ijk', rdun, umuinf)
        F -= np.einsum('i,j...->j...i', rhouinf, dvn)
        F += np.einsum('ij,jk->ijk', p, qnorms)

        return Q, F
        
    def _compute_observer_pfft(self, Q, F):
        omega = self.omega        
        kwv = self.kwv
        mR = self.magR
        mRs = self.magRs
        nR = self.nRvec
        nRs = self.nRsvec
        # fluxes signal windowing
        if self.window:
            Q -= np.mean(Q, 0)
            F -= np.mean(F, 0)
            Q = np.einsum('i,ij->ij', self.wwind, Q)
            F = np.einsum('i,ijk->ijk', self.wwind, F)

        # perform fft of Q, F fluxes
        Qfft = np.empty((self.nfreq, self.nqpts), dtype=np.complex64)
        Ffft = np.empty((self.nfreq, self.nqpts, self.ndims), dtype=np.complex64)
        for iq in range(0, self.nqpts):
            Qfft[:, iq] = self.rfft(Q[:, iq])
            for jd in range(0, self.ndims):
                Ffft[:, iq, jd]  = self.rfft(F[:, iq, jd])

        #compute pfft, i=nob, j=nfreq, k=nqpts, p shape i,j,k 
        kwvR = np.einsum('ik,j->jik', mR, kwv)
        exp_term0 = np.exp(-1j*kwvR)        # exp(-ikR)
        exp_term1 = exp_term0/mRs          # exp(-ikR)/R*
        exp_term2 = exp_term0/(mRs*mRs) # exp(-ikR)/(R* x R*)
        # p1_term
        pfft  = 1j * np.einsum('j,jik,jk->ijk', omega, exp_term1, Qfft)
        # p2_term0 
        pfft += 1j * np.einsum('j,jik,ikm,jkm->ijk', kwv, exp_term1, nR, Ffft)   
        # p2_term1 
        pfft += np.einsum('jik,ikm,jkm->ijk', exp_term2, nRs, Ffft) 

        # surface integration
        pfft = self._surface_integrate(self.qdA, pfft)
        pfft /= (4.*np.pi)
        if self.window:
            pfft *= self.windscale

        return pfft

    def __call__(self, *args, **kwds):
        pass


class PointAcousticSrc(object):
    def __init__(self,name,srclocs,flowinfo,*srcdata):
        self.name = name
        self.srclocs = srclocs
        self.ndims = len(srclocs)
        self.nvars = len(flowinfo)
        self.rhoinf = flowinfo[0]
        self.uinf = flowinfo[1:self.ndims+1]
        self.pinf = flowinfo[-1]

    def exact_solution(self,*args):
        pass
    def update_usoln_onestep(self,*args):
        pass

class MonopoleSrc(PointAcousticSrc):
    def __init__(self, name, srclocs, flowinfo, ampl, srcfreq, ntime, nperiods,
                 gamma):
        super().__init__(name, srclocs, flowinfo, ampl, srcfreq, ntime,
                         nperiods, gamma)
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
        
    def exact_solution(self,xyz_ob):
        nobserv = xyz_ob.shape[0]
        ptime = np.empty((self.nt,nobserv))
        pfft = np.empty((nobserv,self.nfreq), dtype=np.complex128)
        tt =[]
        for i in range(self.nt):
            ti = i*self.dt
            tt.append(ti)
            ptime[i] = self.update_usoln_onestep(xyz_ob, ti, ptime[i], False) 
            - self.pinf
        ptime = np.moveaxis(ptime,0,1)
        for i in range(nobserv):
            pfft[i] = FwhFreqDomainSolver.rfft(ptime[i])
        return self.freq,pfft

    def update_usoln_onestep(self,xyz_ob,tcurr,usoln,uout=True):
        co = np.sqrt(self.pinf/self.rhoinf)
        Mo = self.uinf/co
        kwv = self.omega/co
        xyz_src = self.srclocs
        mR, mRs, nR, nRs = FwhFreqDomainSolver.compute_distance_vars(xyz_src,
         xyz_ob, Mo)
        # phy_potential of the source and its derivatives
        amp_4piRs = self.ampl / (4.*np.pi*mRs)
        phy = amp_4piRs * np.exp(1j * (self.omega*tcurr-kwv*mR)) 
        invRs = 1./mRs
        kk = 1j*kwv*nR + np.einsum('i,ij->ij',invRs,nRs)
        dphy = - np.einsum('i,ij->ji',phy,kk) 
        mdotdphy = sum((Mo*dphy.T).T)

        # computing the flow quantities
        p = np.real(-self.rhoinf * (1j*self.omega*phy + co * mdotdphy)) 
        if not uout:
            return p + self.pinf
        
        usoln[-1] = p + self.pinf
        usoln[0] = p/(co*co) + self.rhoinf  #rho
        usoln[1:self.ndims+1] = np.array([ui + uo 
                                for ui, uo in zip(np.real(dphy),self.uinf)]) #u
        
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
    # Some fnmap face points may need to be flipped to maintain 
    # a c.c.w or c.c. 

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
        ('tet',  20 ):  
            {'tri': [[3, 0, 19], [19, 0, 9], [9, 3, 19], [0, 3, 9]]},
        ('tet',  35 ): 
            {'tri': [[4, 0, 34], [34, 0, 14], [14, 4, 34], [0, 4, 14]]},
        ('hex',  8 ):  
            {'quad': [[0, 1, 3, 2], [0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7],
                     [0, 2, 6, 4], [4, 5, 7, 6]]},
        ('hex',  27 ):  
            {'quad': [[0, 2, 8, 6], [0, 2, 20, 18], [2, 8, 26, 20], 
                     [8, 6, 24, 26], [0, 6, 24, 18], [18, 20, 26, 24]]},
        ('hex',  64 ):  
            {'quad': [[0, 3, 15, 12], [0, 3, 51, 48], [3, 15, 63, 51], 
                     [15, 12, 60, 63], [0, 12, 60, 48], [48, 51, 63, 60]]},
        ('hex',  125 ):  
            {'quad': [[0, 4, 24, 20], [0, 4, 104, 100], [4, 24, 124, 104],
                     [24, 20, 120, 124], [0, 20, 120, 100], 
                     [100, 104, 124, 120]]},
        ('pri',  6 ):  
            {'quad': [[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]], 
             'tri': [[0, 2, 1], [3, 4, 5]]},
        ('pri',  18 ): 
            {'quad': [[0, 2, 14, 12], [2, 5, 17, 14], [0, 12, 17, 5]], 
             'tri': [[0, 5, 2], [12, 14, 17]]},
        ('pri',  40 ):  
            {'quad': [[0, 3, 33, 30], [3, 9, 39, 33], [0, 30, 39, 9]],
             'tri': [[0, 9, 3], [30, 33, 39]]},
        ('pri',  75 ):  
            {'quad': [[0, 4, 64, 60], [4, 14, 74, 64], [0, 60, 74, 14]],
             'tri': [[0, 14, 4], [60, 64, 74]]},
        ('pyr',  5 ):  
            {'quad': [[2, 3, 1, 0]], 'tri': [[0, 1, 4], [1, 3, 4], [3, 2, 4],
                     [0, 4, 2]]},
        ('pyr',  14 ):  
            {'quad': [[6, 8, 2, 0]], 'tri': [[0, 2, 13], [2, 8, 13], 
                     [8, 6, 13], [0, 13, 6]]},
        ('pyr',  30 ): 
            {'quad': [[12, 15, 3, 0]], 'tri': [[0, 3, 29], [3, 15, 29], 
                     [15, 12, 29], [0, 29, 12]]},
        ('pyr',  55 ):  
            {'quad': [[20, 24, 4, 0]], 'tri': [[0, 4, 54], [4, 24, 54], 
                     [24, 20, 54], [0, 54, 20]]},
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
        for (etype, fidx), eidxlist in self._eidxs.items():
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
                comm.Send(self.partsdata[etype], root, tag=53)
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
                    comm.Recv(parr, mrank, tag=53)
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
        sizes.extend(len(varnames)*nb for _, varnames in vvars)

        return npts, ncells, names, types, comps, sizes
