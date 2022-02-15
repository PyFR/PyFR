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

def Gatherv_data_arr(comm, rank, root, data_arr):
    recvbuf = None
    displ = None
    count = None
    sbufsize = 0
    sendbuf = np.empty(0)
    # if np.any(data_arr):
    sbufsize = data_arr.size
    sendbuf = data_arr.reshape(-1)
    count = comm.gather(sbufsize,root=root)

    if rank == root:
        count = np.array(count, dtype='i')
        recvbuf = np.empty(sum(count),dtype=float)
        displ = np.array([sum(count[:p]) for p in range(len(count))])

    comm.Gatherv(sendbuf, [recvbuf, count, displ, MPI.DOUBLE], root=root)

    return recvbuf

def Gather_data_arr(comm, rank, root, data_arr):
    recvbuf = None
    count = None
    sbufsize = data_arr.size
    sendbuf = data_arr.reshape(-1)
    count = comm.gather(sbufsize,root=root)
    if rank == root:
        count = np.array(count, dtype='i')
        recvbuf = np.empty(sum(count),dtype=float)

    comm.Gather(sendbuf, recvbuf, root=root)

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

def Alltoallv_data_arr(incomm, rhsranks, snddata, ncol, sndcnts, rcvcnts, sndperms):
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
    incomm.Alltoallv(s_msg, r_msg)
    
    rhs_rcvsoln = recvbuf.reshape(nrcv, -1) if nrcv else np.empty(0)

    return rhs_rcvsoln


def Ialltoallv_data_arr(incomm, rhsranks, snddata, ncol, sndcnts, rcvcnts, sndperms):
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
    req = incomm.Ialltoallv(s_msg, r_msg)
    
    rhs_rcvsoln = recvbuf.reshape(nrcv, -1) if nrcv else np.empty(0)

    return req, rhs_rcvsoln


TimeParam = namedtuple('TimeParam',['dtsim','dtsub','ltsub','shift','window','psd_scale_mode'])


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

        # Underlying elements class
        self.elementscls = intg.system.elementscls
        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # Extract FWH Surf Geometric Data:
        self._qpts_info, self._m0, _ = self.get_fwhsurf_finfo(elemap, self._eidxs)  #fpts, qpts_dA, norm_pnorms
        self._intqinfo, self._intm0, _ = self.get_fwhsurf_finfo(elemap, self._int_eidxs)
        self._rhsqinfo, self._rhsm0, _ = self.get_fwhsurf_finfo(elemap, self._rhs_eidxs)
        self._rcvqinfo, self._rcvm0, _ = self.get_fwhsurf_finfo(elemap, self._mpi_rcveidxs)
        self._sndqinfo, self._sndm0, _ = self.get_fwhsurf_finfo(elemap, self._mpi_sndeidxs)
        self._bndqinfo, self._bndm0, _ = self.get_fwhsurf_finfo(elemap, self._bnd_eidxs)
        self._mpi_sndcnts, self._mpi_rcvcnts, self._mpi_sndperms, self._mpi_rcvperms = self._prepare_mpiinters_sndrcv_info(elemap)

        self.nintqpts =  np.asarray(self._intqinfo[0]).shape[0]
        self.nrhsqpts = np.asarray(self._rhsqinfo[0]).shape[0]
        self.nrcvqpts = np.asarray(self._rcvqinfo[0]).shape[0]
        self.nsndqpts = np.asarray(self._sndqinfo[0]).shape[0]
        self.nbndqpts = np.asarray(self._bndqinfo[0]).shape[0]
        self.nqpts = np.asarray(self._qpts_info[0]).shape[0]
        self.fwhnvars = len(self.uinf['u'])+2

        qinfo = [[], [], []]
        if self.nintqpts > 0:
            qinfo[0] = self._intqinfo[0]
            qinfo[1] = self._intqinfo[1]
            qinfo[2] = self._intqinfo[2]
        if self.nrcvqpts > 0:
            qinfo[0] = np.vstack((qinfo[0],self._rcvqinfo[0])) if np.any(qinfo[0]) else self._rcvqinfo[0]
            qinfo[1] = np.vstack((qinfo[1],self._rcvqinfo[1])) if np.any(qinfo[1]) else self._rcvqinfo[1]
            qinfo[2] = np.hstack((qinfo[2],self._rcvqinfo[2])) if np.any(qinfo[2]) else self._rcvqinfo[2]
        if self.nbndqpts > 0:
            qinfo[0] = np.vstack((qinfo[0],self._bndqinfo[0])) if np.any(qinfo[0]) else self._bndqinfo[0]
            qinfo[1] = np.vstack((qinfo[1],self._bndqinfo[1])) if np.any(qinfo[1]) else self._bndqinfo[1]
            qinfo[2] = np.hstack((qinfo[2],self._bndqinfo[2])) if np.any(qinfo[2]) else self._bndqinfo[2] 
        self._qpts_info = qinfo

        if self.active_fwhedgrank:
            self._mpi_rhs_fptsplocs = Alltoallv_data_arr(self.fwh_edgecomm, self.fwh_edgranks_list, self._sndqinfo[0], self.ndims,
                                                         self._mpi_sndcnts, self._mpi_rcvcnts, self._mpi_sndperms)

        #Debug, using analytical sources
        analytictest = self.cfg.get(self.cfgsect, 'analytic-src', None)
        self.pacoustsrc = None

        self._started = False
        self._last_prepared = 0.
        # init the fwhsolver and analytical sources if any
        
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

        if self.active_fwhrank or self.active_fwhedgrank:
            # init the fwhsolver
            self.fwhsolver = FwhFreqDomainSolver(timeparam, self.observers, self._qpts_info, self.uinf)
        
        initsoln = None
        restore = False
        if self.active_fwhrank:
            # init from restart
            initsoln = None
            restore = False
            if intg.isrestart:
                restore, initsoln = self.deserialise_root(intg, metadata)
            # Allocate/Init usolution data arrays
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
                    co = np.sqrt(gamma*srcuinf[-1]/srcuinf[0]) 
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
        if intg.tcurr < self.tstart:
            return

        #register globally that we are in the window length
        self._started = True
        
        # if not an active rank return
        if (not self.active_fwhrank) and (not self.active_fwhedgrank):
            return
        
        # If we are not supposed to start fwh yet then return or sampling is not due
        dosample = self.fwhsolver.check_sample(intg.nacptsteps) #intg.nacptsteps % self.samplesteps == 0 
        if not dosample:
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
        fwhroot = self.fwhranks_list[0]
        if fwhroot != wrldroot:
            gatherroot = wrldroot
            gcomm = comm
            grank = rank
        else:
            gatherroot = fwhroot
            gcomm = self.fwh_comm
            grank = self.fwhrank
        
        #compute and gather min fpts-plocs from all other fwh ranks to world root
        #this is to be used for permutation of pfft data
        sendbuf = np.empty(0)
        if self.active_fwhrank:
            idx = range(self.nqpts)
            srtdidx = fuzzysort(self._qpts_info[0].T,idx)
            sendbuf = self._qpts_info[0][srtdidx][0]
        gfplocs_min = Gatherv_data_arr(gcomm, grank, gatherroot, sendbuf)
        #gather pfft of welch averaging data from all other fwh ranks to world root
        sendbuf = np.array([self.fwhsolver.pmagsum, self.fwhsolver.presum, self.fwhsolver.pimgsum]) if self.active_fwhrank else np.empty(0)
        gpfftdata = Gatherv_data_arr(gcomm, grank, gatherroot, sendbuf)

        #gather soln from all other fwh ranks to world root
        sendbuf = self.usoln[:self.fwhsolver.stepcnt].reshape(-1, self.nqpts).T if self.active_fwhrank else np.empty(0)
        gusoln = Gatherv_data_arr(gcomm, grank, gatherroot, sendbuf)

        #gather flux points plocs from all other fwh ranks to world root
        sendbuf = self._qpts_info[0] if self.active_fwhrank else np.empty(0)
        gfptsplocs = Gatherv_data_arr(gcomm, grank, gatherroot, sendbuf)
        
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
                'observers': self.observers,
                'soln'     : gusoln,
                'fptsplocs': gfptsplocs,
                'fptsplocs-min': gfplocs_min,
                'pfftdata' : gpfftdata
            }

        sys.stdout.flush()
        
        return metadata


    def deserialise_root(self, intg, metadata=None):
        usoln = None
        #determine if restore is needed:
        restore = True
        if np.any(metadata['observers'] != self.fwhsolver.xyz_obv):
            print(f'observers locations has changed in the config file, restart cannot be done')
            obb = metadata['observers']
            print(f'written observers {obb}')
            print(f'config  observers {self.fwhsolver.xyz_obv}')
            restore = False
        elif metadata['ntsub'] != self.fwhsolver.ntsub:
            print(f'simulation ntsub {metadata["ntsub"]} (window steps) has changed in the config file ntsub = {self.fwhsolver.ntsub}, restore cannot be done') 
            restore = False
        elif metadata['ltsub'] != self.fwhsolver.ltsub:
            print(f'simulation ltsub {metadata["ltsub"]} (window length) has changed in the config file ntsub = {self.fwhsolver.ltsub}, restore cannot be done') 
            restore = False

        root = 0
        #init gpfftdata first:
        if restore:
            #
            idx = range(self.nqpts)
            srtdidx = fuzzysort(self._qpts_info[0].T, idx)
            sendbuf = self._qpts_info[0][srtdidx][0] 
            recvbuf = None
            if self.fwhrank == root:
                recvbuf = np.empty([self.fwh_comm.size, self.ndims])
            self.fwh_comm.Gather(sendbuf, recvbuf, root=root)
            #curr_fplocs_min = Gatherv_data_arr(self.fwh_comm, self.fwhrank, sendbuf)
            curr_fplocs_min = recvbuf

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
                #
                sendbuf = gpfftdata[srtdidx][perms].astype(float)

            rbufsize = int(metadata['pfftdata'].size/self.fwh_comm.size)
            recvbuf = np.empty(rbufsize, dtype=float)
            self.fwh_comm.Scatter(sendbuf, recvbuf, root=root) 

            recvbuf = recvbuf.reshape(3, -1)
            metadata |= dict(pmagsum=recvbuf[0], presum=recvbuf[1], pimgsum=recvbuf[2])

            #init fwh solver class
            self.fwhsolver.init_from_restart(metadata)

            #init usoln
            self._started = True
            #read metadata
            self._last_prepared = metadata['tlast'] 
            gsoln = metadata['soln']
            gfplocs = metadata['fptsplocs']
            #gather the current global fpts-plocs
            curr_gfplocs = Gatherv_data_arr(self.fwh_comm, self.fwhrank, root, self._qpts_info[0])

            if self.fwhrank == 0:
                nqpts_tot = int(gsoln.shape[0]/(self.fwhsolver.stepcnt*self.fwhnvars))
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
            nsteps = self.fwhsolver.stepcnt 
            rbufsize = int(nsteps * self.fwhnvars * self.nqpts)
            recvbuf = np.empty(rbufsize,dtype=float)
            count = self.fwh_comm.gather(rbufsize,root=0)
            if self.fwhrank == 0:
                sendbuf = gsoln
                count = np.array(count, dtype='i')
                displ = np.array([sum(count[:p]) for p in range(len(count))])

            self.fwh_comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=root)
            usoln = recvbuf.reshape(self.nqpts,-1).T.reshape(nsteps,self.fwhnvars,self.nqpts)

        else:
            self.tstart = intg.tcurr

        return restore, usoln

    
    def deserialise_per_rank(self, intg, metadata=None):
        usoln = None
        restore = self.fwhsolver.init_from_restart(metadata)

        if restore:
            nsteps = self.fwhsolver.stepcnt
            self._started = True
            #read metadata
            self._last_prepared = metadata['tlast'] 
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

        else:
            self.tstart = intg.tcurr

        return restore, usoln


    def _prepare_data(self,intg):
        # Already done for this step
        if self._last_prepared >= intg.tcurr:
            return
        
        # sample solution if it is due
        windowtime = intg.tcurr - self.tstart - (self.fwhsolver.avgcnt-1) * self.fwhsolver.shift * self.fwhsolver.ltsub
        srctime = intg.tcurr - self.tstart
        step = self.fwhsolver.stepcnt
        intqf = self.nintqpts
        rcvq0, rcvqf = intqf, intqf + self.nrcvqpts
        bndq0, bndqf = rcvqf, rcvqf + self.nbndqpts
        intusoln = self.usoln[step, ...,      : intqf]
        rcvusoln = self.usoln[step, ..., rcvq0: rcvqf]
        bndusoln = self.usoln[step, ..., bndq0: bndqf]
        rhsusoln = np.empty((self.fwhnvars, self.nrhsqpts))
        sndusoln = np.empty((self.fwhnvars, self.nsndqpts))
        
        if self.active_fwhedgrank:
            #mpi interfaces with outside elements (rhs)
            if self.nsndqpts:
                if self.pacoustsrc:
                    self.pacoustsrc.update_usoln_onestep(self._sndqinfo[0], srctime, sndusoln)
                else:
                    self.update_usoln_onestep(intg, self._sndm0, self._mpi_sndeidxs, sndusoln)
            #send/recv in an alltoall for averaging over mpi interfaces
            allreq, rhs_rcvsoln = Ialltoallv_data_arr(self.fwh_edgecomm, self.fwh_edgranks_list, 
                                                   sndusoln.T, self.fwhnvars, self._mpi_sndcnts, 
                                                   self._mpi_rcvcnts, self._mpi_sndperms)
            rhs_rcvsoln = rhs_rcvsoln.T

            #mpi interfaces with inside elements (lhs)
            if self.nrcvqpts:
                if self.pacoustsrc:
                    self.pacoustsrc.update_usoln_onestep(self._rcvqinfo[0], srctime, rcvusoln)
                else:
                    self.update_usoln_onestep(intg, self._rcvm0, self._mpi_rcveidxs, rcvusoln)

        if self.pacoustsrc:
            #self.usoln[step] = self.pacoustsrc.update_usoln_onestep(self._qpts_info[0], srctime, self.usoln[step])
            #interior interfaces with inside elements (lhs)
            if self.nintqpts:
                self.pacoustsrc.update_usoln_onestep(self._intqinfo[0], srctime, intusoln)
            #interior interfaces with outside elements (rhs)
            if self.nrhsqpts:
                self.pacoustsrc.update_usoln_onestep(self._rhsqinfo[0], srctime, rhsusoln)
            #boundary interfaces with inside elements 
            if self.nbndqpts:
                self.pacoustsrc.update_usoln_onestep(self._rcvqinfo[0], srctime, bndusoln)
        else:
            #self.usoln[step] = self.update_usoln_onestep(intg, self._m0, self._eidxs, self.usoln[step])
            if self.nintqpts:
                self.update_usoln_onestep(intg, self._intm0, self._int_eidxs, intusoln)
            #interior interfaces with outside elements (rhs)
            if self.nrhsqpts:
                self.update_usoln_onestep(intg, self._rhsm0, self._rhs_eidxs, rhsusoln)
            #boundary interfaces with inside elements 
            if self.nbndqpts:
                self.update_usoln_onestep(intg, self._bndm0, self._bnd_eidxs, bndusoln)
        
        #averaging of solution over interfaces
        if self.nintqpts:
            self._usoln_interface_averageing(self._intqinfo[0], self._rhsqinfo[0], intusoln, rhsusoln)

        if self.active_fwhedgrank:  
            allreq.wait()
            if self.nrcvqpts:
                self._usoln_interface_averageing(self._rcvqinfo[0], self._mpi_rhs_fptsplocs, rcvusoln, rhs_rcvsoln)
        #if self.nbndqpts:
            #self.bndusoln[step] = #apply boundary condition
            
        self.fwhsolver.sampled()
        
        #Debug
        if self.active_fwhrank and self.fwhrank == 0:
            print(f'FWH sampled, wstep {self.fwhsolver.stepcnt-1}, wtime {np.round(windowtime,5)}, srctime {np.round(srctime,5)}, currtime {np.round(intg.tcurr,5)}', flush=True)
            
        # compute fwh solution if due
        docompute = self.fwhsolver.check_compute() 
        if docompute:
            if self.active_fwhrank:
                self.fwhsolver.compute_fwh_solution(self.usoln)
                self.stepcnt = self.fwhsolver.stepcnt
                self.nwindows = self.fwhsolver.avgcnt-1 
                self.avgcnt = self.nwindows+1
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
                    print(f'\nFWH computed (Naver {self.nwindows-1}), ...........................')

                    # Writing spectral results
                    bname = os.path.join(self.basedir,f'{self.basename}'+'_ob{ob}.csv')
                    nwindarr = np.tile(str(self.nwindows),len(pfft[0]))
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

    def _usoln_interface_averageing(self, lfplocs, rfplocs, lsoln, rsoln):
        idx = range(lfplocs.shape[0])
        srtdidx = fuzzysort(lfplocs.T,idx)
        perm = np.argsort(srtdidx)
        idx = range(rfplocs.shape[0])
        srtdidx = fuzzysort(rfplocs.T,idx)
        lsoln += rsoln.T[srtdidx][perm].T
        lsoln *= 0.5
        return lsoln

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
        self._inside_eset  = defaultdict(list)
        self._eidxs        = defaultdict(list)  
        self._int_eidxs    = defaultdict(list)
        self._rhs_eidxs = defaultdict(list)
        self._mpi_rcveidxs = defaultdict(list)  
        self._mpi_sndeidxs = defaultdict(list)
        self._bnd_eidxs    = defaultdict(list) 
        # fwhranks own all fwh edges and hence are the ones that perform acoustic solve
        # fwhedgeranks are ranks who touches an fwh edge but not necessarily
        # owns an edge in general.
        # However, they can be in both fwhedgeranks & fwhranks list if they happen to be both touching 
        # an edge as an outsider and has inside cells (insiders) and hence own some edges
        self.fwhranks_list = []     
        self.fwh_edgranks_list = []

        mesh = intg.system.mesh
        # Collect fwh interfaces and their info
        self.collect_intinters(intg.rallocs, mesh, eset)
        self.collect_mpiinters(intg.rallocs, mesh, eset)
        self.collect_bndinters(intg.rallocs, mesh, eset)

        eidxs = self._eidxs
        self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
        mpiseidxs = self._mpi_sndeidxs
        self._mpi_sndeidxs = {k: np.array(v) for k, v in mpiseidxs.items()}
        mpireidxs = self._mpi_rcveidxs
        self._mpi_rcveidxs = {k: np.array(v) for k, v in mpireidxs.items()}
        bndeidxs = self._bnd_eidxs
        self._bnd_eidxs = {k: np.array(v) for k, v in bndeidxs.items()}


    def collect_intinters(self,rallocs,mesh,eset):
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
                    self._int_eidxs[etype,fidx].append(eidx)
                    self._rhs_eidxs[ifaceR[0],ifaceR[2]].append(ifaceR[1])
                else:
                    etype, eidx, fidx = ifaceR[0:3]
                    self._int_eidxs[etype,fidx].append(eidx)
                    self._rhs_eidxs[ifaceL[0],ifaceL[2]].append(ifaceL[1])

                self._eidxs[etype,fidx].append(eidx)
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)

            # periodic faces:
            elif (flagL & flagR) & (ifaceL[3] != 0 ):   
                etype, eidx, fidx = ifaceL[0:3]
                self._int_eidxs[etype,fidx].append(eidx)
                self._eidxs[etype,fidx].append(eidx)
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)
                #add both right and left faces for vtu writing
                etype, eidx, fidx = ifaceR[0:3]
                self._int_eidxs[etype,fidx].append(eidx)
                self._eidxs[etype,fidx].append(eidx)
                if eidx not in  self._inside_eset[etype]:
                    self._inside_eset[etype].append(eidx)
                self._rhs_eidxs[ifaceL[0],ifaceL[2]].append(ifaceL[1])
                self._rhs_eidxs[ifaceR[0],ifaceR[2]].append(ifaceR[1])

            
    def collect_mpiinters(self, rallocs, mesh, eset):
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
        self._mpi_rcvrank_map = defaultdict(list)
        self._mpi_sndrank_map = defaultdict(list)

        #loop over the mpi interface ranks
        for rhs_prank in rallocs.prankconn[prank]:
            conkey = f'con_p{prank}p{rhs_prank}'
            mpiint = mesh[conkey].astype('U4,i4,i1,i2').tolist()
            flagR = np.empty((len(mpiint)),dtype=bool)
            rhs_mrank = rallocs.pmrankmap[rhs_prank] 
            comm.Recv(flagR,rhs_mrank,tag=52)

            rcnt = 0
            scnt = 0
            #loop over the rank connectivity
            for findex, ifaceL in enumerate(mpiint):

                etype, eidx, fidx = ifaceL[0:3]
                flagL = (eidx in eset[etype]) if etype in eset else False
                # add info if it is an fwh edge, i.e., having either flagL or flagR or both(periodic case) to be True
                if flagL and not flagR[findex] :
                    self._eidxs[etype,fidx].append(eidx)
                    if eidx not in  self._inside_eset[etype]:
                        self._inside_eset[etype].append(eidx)
                    self._mpi_rcveidxs[etype,fidx].append(eidx)
                    self._mpi_rcvrank_map[etype,fidx].append(rhs_mrank)
                    rcnt += 1
                #an interface to be sent to rhs owning rank
                elif flagR[findex] and not flagL:
                    self._mpi_sndeidxs[etype,fidx].append(eidx)
                    self._mpi_sndrank_map[etype,fidx].append(rhs_mrank)
                    scnt += 1

                #periodic mpi interfaces
                elif (flagL and flagR[findex]) and ifaceL[-1] !=0 :
                    self._eidxs[etype,fidx].append(eidx)
                    if eidx not in  self._inside_eset[etype]:
                        self._inside_eset[etype].append(eidx)
                    
                    self._mpi_rcveidxs[etype,fidx].append(eidx)
                    self._mpi_sndeidxs[etype,fidx].append(eidx)
                    self._mpi_rcvrank_map[etype,fidx].append(rhs_mrank)
                    self._mpi_sndrank_map[etype,fidx].append(rhs_mrank)
                    rcnt += 1
                    scnt += 1

            if rcnt or scnt: # this means there is a fwh mpi interface
                if rhs_mrank not in self.fwh_edgranks_list:
                    self.fwh_edgranks_list.append(rhs_mrank)
                if rank not in self.fwh_edgranks_list:
                    self.fwh_edgranks_list.append(rank)


    def _prepare_mpiinters_sndrcv_info(self, elemap):
        #prepare send/recv permutations
        sndperms = defaultdict(list)
        rcvperms = defaultdict(list)
        iq = 0
        for (etype,fidx), eidlist in self._mpi_rcveidxs.items():
            nfpts_perface = elemap[etype].basis.nfacefpts[fidx]
            for id, eid in enumerate(eidlist):
                lrk = np.tile(self._mpi_rcvrank_map[etype,fidx][id], nfpts_perface).tolist()
                for rk in lrk:
                    rcvperms[rk].append(iq)
                    iq += 1 
        iq = 0
        for (etype,fidx), eidlist in self._mpi_sndeidxs.items():
            nfpts_perface = elemap[etype].basis.nfacefpts[fidx]
            for id, eid in enumerate(eidlist):
                lrk = np.tile(self._mpi_sndrank_map[etype,fidx][id], nfpts_perface).tolist()
                for rk in lrk:
                    sndperms[rk].append(iq)
                    iq += 1

        #prepare send/recv counts for mpi edge interfaces
        sndcnts = [0] * len(self.fwh_edgranks_list)
        rcvcnts = [0] * len(self.fwh_edgranks_list)
        for ir, rk in enumerate(self.fwh_edgranks_list):
            sndcnts[ir] = len(sndperms[rk]) if rk in sndperms else 0
            rcvcnts[ir] = len(rcvperms[rk]) if rk in rcvperms else 0

        return sndcnts, rcvcnts, sndperms, rcvperms


    def collect_bndinters(self,rallocs,mesh,eset):
        prank = rallocs.prank
        for f in mesh:
            if (m := re.match(f'bcon_(.+?)_p{prank}$', f)):
                bname = m.group(1)
                bclhs = mesh[f].astype('U4,i4,i1,i2').tolist()
                for ifaceL in bclhs:
                    etype, eidx, fidx = ifaceL[0:3]
                    flagL = (eidx in eset[etype]) if etype in eset else False
                    if flagL:
                        self._bnd_eidxs[etype,fidx].append(eidx)
                        self._eidxs[etype,fidx].append(eidx)
                        if eidx not in  self._inside_eset[etype]:
                            self._inside_eset[etype].append(eidx)


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
    
        #     if self._mpi_rcveidxs:
        #         print(f'\n=================================\nmrank {rank} --> prank {prank} \n=================================')
        #         print(f'FwhEdgeRanks: {self.fwh_edgranks_list}', flush=True)
        #         mranks = []
        #         for rhs_prank in rallocs.prankconn[prank]:
        #             mranks.append(rallocs.pmrankmap[rhs_prank])
        #         print(f'rhs mranks {mranks} --> pranks {rallocs.prankconn[prank]}')
        #         print(f'rhs mranks {mranks_} --> pranks {pranks_}  --- sorted')
            
        #         for (etype,fidx) in self._mpi_rcveidxs:
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

        
        if self.active_fwhrank:
            #prank = intg.rallocs.prank
            #fwhpranks_list = fwh_comm.gather(prank, root=0)

            if self.fwhrank == 0:
                #print(f'\nnum of fwh surface ranks {len(self.fwhranks_list)}')
                print(f'\n{len(self.fwhranks_list)} fwh surface mranks: {self.fwhranks_list}')
                #print(f'{len(self.fwhranks_list)} fwh surface pranks: {fwhpranks_list}\n')
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
        centroids = []

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

            # compute face centroids
            rcpsumdA = 1./np.sum(qdA,1)
            cent = np.einsum('i,ij,ijk->ik', rcpsumdA, qdA, fplocs)
            #xcent = np.array([[c]*nfpts_pertype for c in cent])
            
            dA = qdA.reshape(-1) if nqpts == 0  else np.hstack((dA,qdA.reshape(-1)))
            fpts_plocs = fplocs.reshape(-1,ndims)  if nqpts == 0 else np.vstack((fpts_plocs,fplocs.reshape(-1,ndims) ))
            norm_pnorms = npn.reshape(-1,ndims) if nqpts == 0  else np.vstack((norm_pnorms,npn.reshape(-1,ndims) ))
            centroids = cent if nqpts == 0 else np.vstack((centroids, cent))

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

        return qinfo, m0, centroids    

    def update_usoln_onestep(self, intg, m0_dict, eidxs, usoln):
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
        



class FwhSolverBase(object):
    #if more windows are needed, they can be customally added. A larger list is available in scipy
    windows = {
        'None': (lambda s: np.ones(s), {'density': 1., 'spectrum': 1.}),  # rectangle window
        'hanning'  : (lambda s: np.hanning(s),   {'density': np.sqrt(8./3.), 'spectrum': 2.}),
        'hamming'  : (lambda s: np.hamming(s),   {'density': 50.*np.sqrt(3974.)/1987., 'spectrum': 50./27.}),
        'blackman' : (lambda s: np.blackman(s),  {'density': 50.*np.sqrt(3046.)/1523., 'spectrum': 50./21.}),
        'bartlett' : (lambda s: np.bartlett(s),  {'density': np.sqrt(3.), 'spectrum': 2.}),
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
            raise ValueError(f'ltsub window length {timeparam.ltsub} is too short or less than simulation time step {timeparam.dtsim}')
        if timeparam.shift > 1.:
            raise  ValueError(f'window overlap/shift {timeparam.shift} cannot exceed one, please adjust it as necessary')
        if not timeparam.window in self.windows:
            raise ValueError(f'{timeparam.window} window type is not implemented, please choose ''{None, hanning, hamming, blackman, bartlett}''')

        self._prepare_fft_param(timeparam)

    def _prepare_fft_param(self,timeparam):
        ltsub = timeparam.ltsub
        dtsub = timeparam.dtsub 
        self.shift = shift = timeparam.shift
        self.window = window = timeparam.window if timeparam.window in list(self.windows)[1:] else None
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
        self.samplesteps = self.dtsub/self.dtsim
        
        #(5) window function params
        #since we are using windows for spectral analysis we do not use the last data entry
        self.wwind = self.windows[window][0](ntsub+1)[:-1] if window else window
        self.windscale = self.windows[window][1][scaling_mode] if window else window

    def print_spectraldata(self):
        print(f'\n--------------------------------------')
        print(f'       Adjusted FFT parameters ')
        print(f'--------------------------------------')
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
    def welch_accum(pfft,pmagsum,presum,pimgsum,scale_mode='density'):
        mag = np.abs(pfft)*np.abs(pfft) if scale_mode == 'density' else np.abs(pfft)
        pmagsum += mag
        presum  += np.real(pfft)
        pimgsum += np.imag(pfft)
        return pmagsum,presum,pimgsum

    @staticmethod
    def welch_average(nwindows,pmagsum,presum,pimgsum,scale_mode='density'):
        mag = np.sqrt(pmagsum/nwindows) if scale_mode=='density' else pmagsum/nwindows
        phase = np.arctan2(pimgsum,presum)
        pfft = mag * np.exp(1j * phase)
        return pfft

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

    def __init__(self, timeparm, observers, surfdata, Uinf, surftype='permeable'):
        super().__init__(timeparm, observers, surfdata, Uinf, surftype)
        
        # compute distance vars for fwh
        if np.any(self.xyz_src):
            self.magR, self.magRs, self.nRvec, self.nRsvec = self._compute_distance_vars(self.xyz_src,self.xyz_obv,self.uinf['Mach'])
        # compute frequency parameters
        self.freq  = np.fft.rfftfreq(self.ntsub,self.dtsub)
        self.omega = 2*np.pi*self.freq
        self.kwv = self.omega/self.uinf['c']
        self.nfreq = np.size(self.freq)

        # Init fwh outputs 
        self.pfft = np.zeros((self.nobserv,self.nfreq),dtype=np.complex128)
        self.pmagsum = np.zeros((self.nobserv,self.nfreq))
        self.presum = np.zeros((self.nobserv,self.nfreq))
        self.pimgsum = np.zeros((self.nobserv,self.nfreq))
    
    def init_from_restart(self, initdata):
        self.stepcnt = initdata['stepcnt']
        self.avgcnt  = initdata['avgcnt']
        self.ntsub   = initdata['ntsub']
        self.pmagsum = initdata['pmagsum'].reshape(self.nobserv, -1)
        self.presum  = initdata['presum'].reshape(self.nobserv, -1)
        self.pimgsum = initdata['pimgsum'].reshape(self.nobserv, -1)

        return


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

    def compute_fwh_solution(self, usoln):
        #compute fluxes in time domain
        Q, F = self._compute_surface_fluxes(usoln)
        #compute fwh pressure solution in frequency domain
        self.pfft = self._compute_observer_pfft(Q,F)
        # use welch method if averaging
        if self.averaging:
            self.pmagsum, self.presum, self.pimgsum = self.welch_accum(self.pfft, self.pmagsum, self.presum, self.pimgsum, self.psd_scale_mode)
            if self.avgcnt > 1:
                self.pfft = self.welch_average(self.avgcnt, self.pmagsum, self.presum, self.pimgsum, self.psd_scale_mode)
            #update the averaging counter
            self.avgcnt += 1
        #update the step counter
        self.stepcnt = self.ntsub - self.ntoverlap

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
        rdun = rho_tot*dUn
        umuinf = u-uinf
        rhouinf = rhoinf*uinf
        Q = rdun + rhoinf*dvn
        F = np.einsum('ij,ijk->ijk',rdun,umuinf)
        F -= np.einsum('i,j...->j...i',rhouinf,dvn)
        F += np.einsum('ij,jk->ijk',p,qnorms)

        return Q,F
        
    def _compute_observer_pfft(self,Q,F):        
        # fluxes signal windowing
        if self.window:
            Q -= np.mean(Q,0)
            F -= np.mean(F,0)
            Q = np.einsum('i,ij->ij',self.wwind,Q)
            F = np.einsum('i,ijk->ijk',self.wwind,F)

        # perform fft of Q, F fluxes
        Qfft = np.empty((self.nfreq,self.nqpts),dtype=np.complex64)
        Ffft = np.empty((self.nfreq,self.nqpts,self.ndims),dtype=np.complex64)
        for iq in range(0,self.nqpts):
            Qfft[:,iq] = self.rfft(Q[:,iq])
            for jd in range(0,self.ndims):
                Ffft[:,iq,jd]  = self.rfft(F[:,iq,jd])

        #compute pfft, i=nob,j=nfreq,k=nqpts, p shape i,j,k 
        kwvR = np.einsum('ik,j->jik',self.magR,self.kwv)
        exp_term0 = np.exp(-1j*kwvR)        # exp(-ikR)
        exp_term1 = exp_term0/self.magRs          # exp(-ikR)/R*
        exp_term2 = exp_term0/(self.magRs*self.magRs) # exp(-ikR)/(R*xR*)
        pfft  = 1j * np.einsum('j,jik,jk->ijk',self.omega, exp_term1, Qfft)        # p1_term
        pfft += 1j * np.einsum('j,jik,ikm,jkm->ijk',self.kwv,exp_term1,self.nRvec,Ffft)   # p2_term0
        pfft += np.einsum('jik,ikm,jkm->ijk',exp_term2,self.nRsvec,Ffft)                  # p2_term1 

        # surface integration
        pfft = self._surface_integrate(self.qdA,pfft) / (4.*np.pi)
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
        
    def exact_solution(self,xyz_ob):
        nobserv = xyz_ob.shape[0]
        ptime = np.empty((self.nt,nobserv))
        pfft = np.empty((nobserv,self.nfreq),dtype=np.complex128)
        tt =[]
        for i in range(self.nt):
            tt.append(i*self.dt)
            ptime[i] = self.update_usoln_onestep(xyz_ob,i*self.dt,ptime[i],False) - self.pinf
        ptime = np.moveaxis(ptime,0,1)
        for i in range(nobserv):
            pfft[i] = FwhFreqDomainSolver.rfft(ptime[i])
        return self.freq,pfft

    def update_usoln_onestep(self,xyz_ob,tcurr,usoln,uout=True):
        co = np.sqrt(self.gamma*self.pinf/self.rhoinf)
        Mo = self.uinf/co
        kwv = self.omega/co
        magR, magRs, R_nvec, Rs_nvec = FwhFreqDomainSolver._compute_distance_vars(self.srclocs,xyz_ob,Mo)
        # phy_potential of the source and its derivatives
        phy = (self.ampl/(4.*np.pi*magRs))*np.exp(1j*(self.omega*tcurr-kwv*magR)) 
        invRs = 1./magRs
        kk = 1j*kwv*R_nvec + np.einsum('i,ij->ij',invRs,Rs_nvec)
        dphy = - np.einsum('i,ij->ji',phy,kk) 
        mdotdphy = sum((Mo*dphy.T).T)

        # computing the flow quantities
        p = np.real(-self.rhoinf * (1j*self.omega*phy + co * mdotdphy)) 
        if not uout:
            return p + self.pinf
        
        usoln[-1] = p + self.pinf
        usoln[0] = p/(co*co) + self.rhoinf  #rho
        usoln[1:self.ndims+1] = np.array([ui + uo for ui, uo in zip(np.real(dphy),self.uinf)])  #u
        
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
        sizes.extend(len(varnames)*nb for _ , varnames in vvars)

        return npts, ncells, names, types, comps, sizes
