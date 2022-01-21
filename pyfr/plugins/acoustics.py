# -*- coding: utf-8 -*-

from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
import os, sys
import re
import numpy as np
from collections import defaultdict
from mpi4py import MPI

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class FwhAcousticsPlugin(PostactionMixin, BasePlugin):

    name = 'fwhacoustics'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.DEBUG = self.cfg.getint(cfgsect,'debug',0)
        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')
        
        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]
        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Check if the system is incompressible
        self._ac = intg.system.name.startswith('ac')

        # Region of interest
        box = self.cfg.getliteral(self.cfgsect, 'region')
        # Prepare fwh surface mesh data
        region_eset = self.extract_drum_region_eset(intg, box)
        self.prepare_surfmesh(intg,region_eset)

        # Determine fwh active ranks lists
        self.fwh_comm, self.fwh_edgecomm = self.build_fwh_subranks_lists()        

        # Construct the fwh mesh writer
        self._writer = NativeWriter(intg, basedir, basename, 'soln')

        # write the fwh surface geometry file
        self._write_fwh_surface_geo(intg,self._eidxs)

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then make sure we write out the initial
        # solution when we are called for the first time
        if not intg.isrestart:
            self.tout_last -= self.dt_out

        # Underlying elements class
        self.elementscls = intg.system.elementscls
        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # Extract FWH Surf Geometric Data:
        self._m0, self._qwts, self._norms, self._xfpts = self.get_fwh_surfdata(elemap, self._eidxs)


    def __call__(self, intg):
        # Return if no output is due
        # if intg.nacptsteps % self.nsteps:
        #     return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))
        ndims, nvars = self.ndims, self.nvars

        for etype, fidx in self._m0:
            # Get the interpolation operator
            m0 = self._m0[etype, fidx]
            nfpts, nupts = m0.shape

            # Extract the relevant elements from the solution
            uupts = solns[etype][..., self._eidxs[etype, fidx]]

            #print(f'nfpts: {nfpts} , nupts: {nupts}')
            #print(f'uupts[{etype},{fidx}].shape = {uupts.shape}')

            # Interpolate to the face
            ufpts = m0 @ uupts.reshape(nupts, -1)
            ufpts = ufpts.reshape(nfpts, nvars, -1)
            ufpts = ufpts.swapaxes(0, 1)

            #print(f'ufpts[{etype},{fidx}].shape = {ufpts.shape}')

            # Compute the pressure
            pidx = 0 if self._ac else -1
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[pidx]

            #print(f'p[{etype},{fidx}].shape = {p.shape}')

        # If a post-action has been registered then invoke it
        # self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
        #                         soln=self.fwhmeshfname, t=intg.tcurr)
        
        # Update the last output time
        self.tout_last = intg.tcurr

        # Need to print fwh surface geometric data only once
        intg.completed_step_handlers.remove(self)

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

    def _write_fwh_surface_geo(self,intg,eset):
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
        for (etype,fidx), eidxs in sorted(eset.items()):
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
        vtuwriter = VTUSurfWriter(intg,self._eidxs)
        # Write VTU file:
        vtumfname = os.path.splitext(self.fwhmeshfname)[0]
        vtuwriter._write(vtumfname)

    def prepare_surfmesh(self,intg,eset):
        self._eidxs= defaultdict(list)  # fwh face pairs set
        self._intinters = defaultdict(list)
        self._mpiinters = defaultdict(list)
        self._rhsmpiinters = defaultdict(list)
        self._bndinters = defaultdict(list)
        self.nintinters = {}
        self.nmpiinters = {}
        self.nbndinters = {}
        self.ntotinters = {}
        # fwhranks own all fwh edges and hence are the ones that perform acoustic solve
        # fwhedgeranks are ranks who touches an fwh edge but not necessarily
        # owns an edge in general.
        # However, they can be in both fwhedgeranks & fwhranks list if they happen to be both touching 
        # an edge as an outsider and has inside cells (insiders) and hence own some edges
        self.fwhranks_list = []     
        self.fwh_edgranks_list = []

        mesh = intg.system.mesh
        # Collect fwh interfaces and their info
        self.collect_intinters(intg.rallocs,mesh,eset)
        self.collect_mpiinters(intg.rallocs,mesh,eset)
        self.collect_bndinters(intg.rallocs,mesh,eset)

        for (etype,fidx) in self._eidxs:
            if (etype,fidx) in self.nintinters:
                self.ntotinters[etype,fidx] = self.nintinters[etype,fidx] 
            if (etype,fidx) in self.nmpiinters:
                if (etype,fidx) in self.ntotinters:
                    self.ntotinters[etype,fidx] += self.nmpiinters[etype,fidx]
                else:
                    self.ntotinters[etype,fidx] = self.nmpiinters[etype,fidx]
            if (etype,fidx) in self.nbndinters:
                if (etype,fidx) in self.ntotinters:
                    self.ntotinters[etype,fidx] += self.nbndinters[etype,fidx] 
                else:
                    self.ntotinters[etype,fidx] = self.nbndinters[etype,fidx]

        eidxs = self._eidxs
        self._eidxs = {k: np.array(v) for k, v in eidxs.items()}

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
                    self._intinters[etype,fidx].append(ifaceR[0:3])
                    self._eidxs[etype,fidx].append(eidx)
                else:
                    etype, eidx, fidx = ifaceR[0:3]
                    self._intinters[etype,fidx].append(ifaceL[0:3])
                    self._eidxs[etype,fidx].append(eidx)
            # periodic faces:
            elif (flagL & flagR) & (ifaceL[3] != 0 ):   
                etype, eidx, fidx = ifaceL[0:3]
                self._intinters[etype,fidx].append(ifaceR[0:3])
                self._eidxs[etype,fidx].append(eidx)
                #add both right and left faces for vtu writing
                etype, eidx, fidx = ifaceR[0:3]
                self._intinters[etype,fidx].append(ifaceL[0:3])
                self._eidxs[etype,fidx].append(eidx) 

        for (etype,fidx) in self._intinters:
            self.nintinters[etype,fidx] = len(self._intinters[etype,fidx])

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
            self._rhsmpiinters[etype,fidx].append((findex,rhs_mrank))
            comm.Send(flagL,rhs_mrank,tag=52)

        # receive flags and collect mpi interfaces
        for rhs_prank in rallocs.prankconn[prank]:
            conkey = f'con_p{prank}p{rhs_prank}'
            mpiint = mesh[conkey].astype('U4,i4,i1,i2').tolist()
            flagR = np.empty((len(mpiint)),dtype=bool)
            rhs_mrank = rallocs.pmrankmap[rhs_prank] 
            comm.Recv(flagR,rhs_mrank,tag=52)

            for findex, ifaceL in enumerate(mpiint):
                etype, eidx, fidx = ifaceL[0:3]
                flagL = (eidx in eset[etype]) if etype in eset else False
                # add info if it is an fwh edge
                if flagL and not flagR[findex] :
                    self._mpiinters[etype,fidx].append((findex,rhs_mrank))
                    self._eidxs[etype,fidx].append(eidx)
                    
                if (not (flagL & flagR[findex])) & (flagL | flagR[findex]):
                    if rhs_mrank not in self.fwh_edgranks_list:
                        self.fwh_edgranks_list.append(rhs_mrank)
                    if rank not in self.fwh_edgranks_list:
                        self.fwh_edgranks_list.append(rank)

        for (etype,fidx) in self._mpiinters:
            self.nmpiinters[etype,fidx] = len(self._mpiinters[etype,fidx])

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
                        self._eidxs[etype,fidx].append(eidx)

        for (etype,fidx) in self._bndinters:
            self.nbndinters[etype,fidx] = len(self._bndinters[etype,fidx])

    def build_fwh_subranks_lists(self):
        self.fwhranks_list = []
        comm, rank, root = get_comm_rank_root()

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

        if rank==root:
            print(f'FwhCompRanks: {self.fwhranks_list}', flush=True)
            print(f'FwhEdgeRanks: {self.fwh_edgranks_list}', flush=True)

        return fwh_comm, fwh_edgecomm

    def get_fwh_surfdata(self, elemap, eidxs):
        # Interpolation matrices and quadrature weights
        m0 = {}
        qwts = defaultdict(list)
        norms = defaultdict(list)
        fpts = defaultdict(list)

        for (etype,fidx), eidlist in eidxs.items():
            eles = elemap[etype]
            
            if (etype, fidx) not in m0:
                facefpts = eles.basis.facefpts[fidx]
                m0[etype, fidx] = eles.basis.m0[facefpts]
                qwts[etype, fidx] = eles.basis.fpts_wts[facefpts]
            
            for eidx in eidlist:
                fpts[etype, fidx].append(eles.plocfpts[facefpts, eidx])
                
                # Unit physical normals and their magnitudes (including |J|)
                npn = eles.get_norm_pnorms(eidx, fidx)
                mpn = eles.get_mag_pnorms(eidx, fidx)
                norms[etype, fidx].append(mpn[:, None]*npn)


        norms_arr = {k: np.array(v) for k, v in norms.items()}
        xfpts = {k: np.array(v) for k, v in fpts.items()}

        return m0, qwts, norms_arr, xfpts      



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
    # reverse map from face index to face type
    fnum_pftype_map = {
            'tri' : { 0: 'line', 1: 'line', 2: 'line'}, 
            'quad': { 0: 'line', 1: 'line', 2: 'line', 3: 'line'}, 
            'tet' : { 0: 'tri', 1: 'tri', 2: 'tri', 3: 'tri'}, 
            'hex' : { 0: 'quad', 1: 'quad', 2: 'quad', 3: 'quad', 4: 'quad', 5: 'quad'},
            'pri' : { 0: 'tri', 1: 'tri', 2: 'quad', 3: 'quad', 4: 'quad'}, 
            'pyr' : { 0: 'quad', 1: 'tri', 2: 'tri', 3: 'tri', 4: 'tri'} 
    }
    # offset to get local fidx/fnum inside each facetype map
    _fnum_offset = {'tri' : {'line': 0},
                    'quad': {'line': 0},
                    'tet' : {'tri': 0},
                    'hex' : {'quad': 0},
                    'pri' : {'quad': -2, 'tri': 0}, 
                    'pyr' : {'quad': 0, 'tri': -1}}

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

        self.prepare_vtufnodes()

    def _write(self,fname):
        fname = f'{fname}.vtu'
        self.write_vtu_out(fname)

    def prepare_vtufnodes(self): 
        for (etype,fidx), eidxlist in self._eidxs.items():
            pts = np.swapaxes(self.mesh[f'spt_{etype}_p{self.rallocs.prank}'],0,1)
            pftype = self.fnum_pftype_map[etype][fidx]
            fidx = np.int(fidx) + self._fnum_offset[etype][pftype]
            for eidx in eidxlist:
                #eidx = np.int(eidx)
                nelemnodes = pts[eidx].shape[0]
                nidx = self._petype_fnmap[etype,nelemnodes][pftype][fidx]
                self._vtufnodes[pftype].append(pts[eidx][nidx,:])

    def prepare_vtufnodes_info(self):
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

    def write_vtu_out(self,fname):

        comm, rank, root = get_comm_rank_root()
        # prepare nodes info for each rank
        info = self.prepare_vtufnodes_info()

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
        exit(0)

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
