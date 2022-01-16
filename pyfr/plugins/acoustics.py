# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from collections import defaultdict

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter

class AcousticsPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'acoustics'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the solution writer
        self._writer = NativeWriter(intg, basedir, basename, 'soln')

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then make sure we write out the initial
        # solution when we are called for the first time
        if not intg.isrestart:
            self.tout_last -= self.dt_out

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

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

        # Fetch data from other plugins and add it to metadata with ad-hoc keys
        for csh in intg.completed_step_handlers:
            try:
                prefix = intg.get_plugin_data_prefix(csh.name, csh.suffix)
                pdata = csh.serialise(intg)
            except AttributeError:
                pdata = {}

            if rank == root:
                metadata.update({f'{prefix}/{k}': v for k, v in pdata.items()})

        # Fetch and (if necessary) subset the solution
        data = dict(self._ele_region_data)
        for idx, etype, rgn in self._ele_regions:
            data[etype] = intg.soln[idx][..., rgn].astype(self.fpdtype)

        # Write out the file
        solnfname = self._writer.write(data, intg.tcurr, metadata)

        # If a post-action has been registered then invoke it
        self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                soln=solnfname, t=intg.tcurr)

        # Update the last output time
        self.tout_last = intg.tcurr


class FwhSurfWriterPlugin(PostactionMixin, RegionMixin, BasePlugin):

    DEBUG = 0
    name = 'fwhsurfwriter'
    systems = ['*']
    formulations = ['dual', 'std']

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
    _fnum_pftype_map = {
            'tri' : [(0, 'line'), (1, 'line'), (2, 'line')], 
            'quad': [(0, 'line'), (1, 'line'), (2, 'line'), (3, 'line')], 
            'tet' : [(0, 'tri'), (1, 'tri'), (2, 'tri'), (3, 'tri')], 
            'hex' : [(0, 'quad'), (1, 'quad'), (2, 'quad'), (3, 'quad'), (4, 'quad'), (5, 'quad')], 
            'pri' : [(0, 'tri'), (1, 'tri'), (2, 'quad'), (3, 'quad'), (4, 'quad')], 
            'pyr' : [(0, 'quad'), (1, 'tri'), (2, 'tri'), (3, 'tri'), (4, 'tri')]
    }

    # offset to get local fidx/fnum inside each facetype map
    _fnum_offset = {'tri' : {'line': 0},
                    'quad': {'line': 0},
                    'tet' : {'tri': 0},
                    'hex' : {'quad': 0},
                    'pri' : {'quad': -2, 'tri': 0}, 
                    'pyr' : {'quad': 0, 'tri': -1}}

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.DEBUG = self.cfg.getint(cfgsect,'debug',0)
        
        # prepare fwh surface mesh data
        self.prepare_surfmesh(intg,self.region_eset)

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the solution writer
        self._writer = NativeWriter(intg, basedir, basename, 'soln')

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then make sure we write out the initial
        # solution when we are called for the first time
        if not intg.isrestart:
            self.tout_last -= self.dt_out

    def __call__(self, intg):

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

        # Fetch data from other plugins and add it to metadata with ad-hoc keys
        for csh in intg.completed_step_handlers:
            try:
                prefix = intg.get_plugin_data_prefix(csh.name, csh.suffix)
                pdata = csh.serialise(intg)
            except AttributeError:
                pdata = {}

            if rank == root:
                metadata.update({f'{prefix}/{k}': v for k, v in pdata.items()})

        # Fetch and (if necessary) subset the solution
        data = dict(self._ele_region_data)
        for idx, etype, rgn in self._ele_regions:
            data[etype] = intg.soln[idx][..., rgn].astype(self.fpdtype)

        # Write out the file
        solnfname = self._writer.write(data, intg.tcurr, metadata)
        name = os.path.splitext(solnfname)[0]
        self.write_vtu_out(f'{name}.vtu')
        if self.DEBUG:
           self.write_vtk_out(f'{name}.vtk','unstructured')

        # If a post-action has been registered then invoke it
        self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                soln=solnfname, t=intg.tcurr)

        # Update the last output time
        self.tout_last = intg.tcurr

        # Need to print fwh surface geometric data only once
        intg.completed_step_handlers.remove(self)


    def prepare_surfmesh(self,intg,eset):
        pts = {}
        self.fwheset = defaultdict(list) # fwh surface 3D elements ids set
        self.fwhelespts = {}               # fwh 3D elements to nodes set
        self.fwhfset = defaultdict(list)    # fwh face pairs set
        self.vtufnodes = defaultdict(list)    # fwh face to nodes set
        self.fwhranks = []                    # active ranks for fwh

        comm, rank, root = get_comm_rank_root()
        prank = intg.rallocs.prank
        mesh = intg.system.mesh
        
        # Collect elements pts from the mesh
        for etype in eset:
            pts[etype] = np.swapaxes(mesh[f'spt_{etype}_p{prank}'],0,1)

        # Collect fwh interfaces and their info
        self.collect_intinters(prank,mesh,pts,eset)
        self.collect_mpiinters(comm,intg.rallocs,mesh,pts,eset)
        self.collect_bndinters(prank,mesh,pts,eset)

        # determine active ranks for fwh
        self.isfwhactive = False
        if self.fwhfset:
            self.isfwhactive = True
        rank_is_active_list = comm.gather(self.isfwhactive,root=root)
        
        if rank==root:
            for i, flag in enumerate(rank_is_active_list):
                if flag:
                    self.fwhranks.append(i)

        self.fwhranks=comm.bcast(self.fwhranks,root=root)
        
        # Prepare face VTU nodes:
        self.prepare_vtufnodes()

    def collect_intinters(self,prank,mesh,pts,eset):
        flhs, frhs = mesh[f'con_p{prank}'].astype('U4,i4,i1,i2').tolist()
        for ifaceL,ifaceR in zip(flhs,frhs):    
            etype,eidx = ifaceR[0:2]
            flagR = (eidx in eset[etype]) if etype in eset else False
            etype,eidx = ifaceL[0:2]
            flagL = (eidx in eset[etype]) if etype in eset else False
            # interior faces
            if (not (flagL & flagR)) & (flagL | flagR):
                pftype = self._get_pftype_from_fpairs(ifaceL,ifaceR)
                if flagL:
                    self.fwhfset[pftype].append([ifaceL,ifaceR])
                else:
                    etype, eidx = ifaceR[0:2]
                    # ensure we are using the element inside 
                    # the region as the left element
                    self.fwhfset[pftype].append([ifaceR,ifaceL])
                self.fwheset[etype].append(eidx)
                self.fwhelespts[etype,eidx] = pts[etype][eidx,:,:]
            # periodic faces:
            elif (flagL & flagR) & (ifaceL[3] != 0 ): 
                pftype = self._get_pftype_from_fpairs(ifaceL,ifaceR)
                self.fwhfset[pftype].append([ifaceL,ifaceR])
                self.fwhfset[pftype].append([ifaceR,ifaceL])  
                
                etype, eidx = ifaceL[0:2]
                self.fwheset[etype].append(eidx)
                self.fwhelespts[etype,eidx] = pts[etype][eidx,:,:]
                etype, eidx = ifaceR[0:2]
                self.fwheset[etype].append(eidx)
                self.fwhelespts[etype,eidx] = pts[etype][eidx,:,:]

    def collect_mpiinters(self,comm,rallocs,mesh,pts,eset):
        prank = rallocs.prank
        # send flags
        for rhs_prank in rallocs.prankconn[prank]:
            mpiint = mesh[f'con_p{prank}p{rhs_prank}'].astype('U4,i4,i1,i2').tolist()
            flagL = np.zeros((len(mpiint)),dtype=bool)
            for ii, ifaceL in enumerate(mpiint):
                etype, eidx, fidx = ifaceL[0:3]
                flagL[ii] = (eidx in eset[etype]) if etype in eset else False
            rhs_mrank = rallocs.pmrankmap[rhs_prank]
            comm.Send(flagL,rhs_mrank,tag=52)

        # receive flags and collect mpi interfaces    
        for rhs_prank in rallocs.prankconn[prank]:
            conkey = f'con_p{prank}p{rhs_prank}'
            mpiint = mesh[conkey].astype('U4,i4,i1,i2').tolist()
            flagR = np.empty((len(mpiint)),dtype=bool)
            rhs_mrank = rallocs.pmrankmap[rhs_prank]
            comm.Recv(flagR,rhs_mrank,tag=52)

            ifaceR = [conkey,-1e20,-1,rhs_mrank]
            for ii, ifaceL in enumerate(mpiint):
                etype, eidx, fidx = ifaceL[0:3]
                flagL = (eidx in eset[etype]) if etype in eset else False
                if (not (flagL & flagR[ii])) & flagL :
                    pftype = self._fnum_to_pftype(etype,fidx)
                    self.fwhfset[pftype].append([ifaceL,ifaceR])
                    self.fwheset[etype].append(eidx)
                    self.fwhelespts[etype,eidx] = pts[etype][eidx,:,:]

    def collect_bndinters(self,prank,mesh,pts,eset):
        for f in mesh:
            if (m := re.match(f'bcon_(.+?)_p{prank}$', f)):
                bname = m.group(1)
                bclhs = mesh[f].astype('U4,i4,i1,i2').tolist()
                ifaceR = [f'bcon_{bname}_p{prank}',-1e20,-1,-1e20]
                for ifaceL in bclhs:
                    etype, eidx, fidx = ifaceL[0:3]
                    flagL = (eidx in eset[etype]) if etype in eset else False
                    if flagL:
                        pftype = self._fnum_to_pftype(etype,fidx)
                        self.fwhfset[pftype].append([ifaceL,ifaceR])
                        self.fwheset[etype].append(eidx)
                        self.fwhelespts[etype,eidx] = pts[etype][eidx,:,:]

    def prepare_vtufnodes(self):
        for pftype, fpairs in self.fwhfset.items():
            flhs = np.moveaxis(fpairs,0,1)[0]
            for ifaceL in flhs:
                etype,eidx,fidx = ifaceL[0:3]
                eidx = np.int(eidx)
                fidx = np.int(fidx) + self._fnum_offset[etype][pftype] 
                nelnodes = self.fwhelespts[etype,eidx].shape[0]
                nidx = self._petype_fnmap[etype,nelnodes][pftype][fidx]
                self.vtufnodes[pftype].append(self.fwhelespts[etype,eidx][nidx,:])

    def write_vtu_out(self,fname):

        comm, rank, root = get_comm_rank_root()
        # prepare nodes info for each rank
        info = self.prepare_vtufnodes_info(self.vtufnodes)

        # Communicate and prepare data for writing
        if rank != root:
            # Send the info about our data points to the root rank
            comm.gather(info, root=root)
            # Send the data points itself
            for etype,arrs in self.vtufnodes.items():
                comm.Send(np.array(arrs).astype(info[etype]['dtype']), root)
        #root
        else:
            # Collect info about what remote ranks want to write 
            ginfo = comm.gather({}, root)
            # Update the info and receive the node arrays
            vpts_global = {}
            # root nodes first
            for etype in info:
                vpts_global[etype] = np.array(self.vtufnodes[etype])

            # update info and receive/stack nodes from other ranks
            for mrank, minfo in enumerate(ginfo):
                for etype, vals in minfo.items():
                    if etype in info:
                        info[etype]['vtu_attr'][3] = [sum(x) for x in zip(info[etype]['vtu_attr'][3], vals['vtu_attr'][3])]
                        info[etype]['mesh_attr'] = [sum(x) for x in zip(info[etype]['mesh_attr'],vals['mesh_attr'])]
                        shapes = [x for x in info[etype]['shape']]
                        shapes[0] += vals['shape'][0]
                        info[etype]['shape'] = tuple(shapes)
                    else:
                        info[etype] = vals
                    varr = np.empty(vals['shape'], dtype=vals['dtype'])
                    comm.Recv(varr, mrank)
                    vpts_global[etype] = np.vstack((vpts_global[etype],varr)) if etype in vpts_global else varr

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
                    self._write_data(fh, etype, vpts_global[etype])

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

    def _write_data(self, vtuf, mk, vpts):
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


    def _get_pftype_from_fpairs(self,ifaceL,ifaceR):
        pftype0 = self._fnum_to_pftype(ifaceL[0],ifaceL[2])
        pftype1 = self._fnum_to_pftype(ifaceR[0],ifaceR[2])
        if pftype0 != pftype1:
            raise KeyError(f'pftypeL: {pftype0} is not equal to pftypeR: {pftype1}')
        else:
           return pftype0

    def _fnum_to_pftype(self,inetype,fnum):
        return self._fnum_pftype_map[inetype][fnum][1]

    def _get_npts_ncells(self,mk): 
        ncells = np.asarray(self.vtufnodes[mk]).shape[0]
        npts = ncells * self._petype_focount_map[mk]
        return npts, ncells

    def _get_array_attrs(self,mk=None):
        fpdtype = self.fpdtype 
        vdtype = 'Float32' if fpdtype == np.float32 else 'Float64'
        dsize = np.dtype(fpdtype).itemsize 

        names = ['', 'connectivity', 'offsets', 'types']
        types = [vdtype, 'Int32', 'Int32', 'UInt8']
        comps = ['3', '', '', '']

        if mk:
            npts, ncells = self._get_npts_ncells(mk)
            nb = npts*dsize
            sizes = [3*nb, 4*npts, 4*ncells, ncells]

            return names, types, comps, sizes
        else:
            return names, types, comps

    def prepare_vtufnodes_info(self, fnodes):
        info = defaultdict(dict)
        for k, v in fnodes.items():
            names, types, comps, sizes = self._get_array_attrs(k)
            npts, ncells = self._get_npts_ncells(k)
            info[k]['vtu_attr'] = [names, types, comps, sizes]
            info[k]['mesh_attr'] = [npts, ncells]
            info[k]['shape'] = np.asarray(v).shape 
            info[k]['dtype'] = np.asarray(v).dtype.str
        return info
