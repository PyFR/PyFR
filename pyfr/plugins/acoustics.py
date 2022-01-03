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
        ('tri',  3 ):  {'line': [[2, 0], [0, 1], [1, 2]]},
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

        self.DEBUG = self.cfg.getint(cfgsect,'debug')
        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')
        self.fwhfname = os.path.join(basedir,basename)

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

        # prepare fwh surface mesh data
        self._prepare_surfmesh(intg,self.region_eset)

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
        self.write_vtu_out(self.fwhfname+".vtu")
        if self.DEBUG:
           self.write_vtk_out(self.fwhfname+".vtk",'unstructured')

        # If a post-action has been registered then invoke it
        self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                soln=solnfname, t=intg.tcurr)

        # Update the last output time
        self.tout_last = intg.tcurr

        # Need to print fwh surface geometric data only once
        intg.completed_step_handlers.remove(self)


    def _prepare_surfmesh(self,intg,eset):
        pts = {}
        feleset = defaultdict(list) # fwh surface 3D elements ids set
        felespts = {}               # fwh 3D elements to nodes set
        fset = defaultdict(list)    # fwh face pairs set
        nset = defaultdict(list)    # fwh face to nodes set

        mesh = intg.system.mesh
        for etype in intg.system.ele_types:
            pts[etype] = np.swapaxes(mesh[f'spt_{etype}_p{intg.rallocs.prank}'],0,1)

        key = f'con_p{intg.rallocs.prank}'
        flhs, frhs = mesh[key].astype('U4,i4,i1,i2').tolist()

        # Collect interior interfaces of the fwh surf and their info
        etypes = list(eset)
        for ifaceL,ifaceR in zip(flhs,frhs):
            
            etype,eidx = ifaceR[0:2]
            flagR = (eidx in eset[etype]) if etype in etypes else False

            etype,eidx = ifaceL[0:2]
            flagL = (eidx in eset[etype]) if etype in etypes else False

            if (not (flagL & flagR)) & (flagL | flagR):
                pftype = self._get_pftype_from_fpairs(ifaceL,ifaceR)
                if flagL:
                    fset[pftype].append([ifaceL,ifaceR])
                else:
                    etype, eidx = ifaceR[0:2]
                    fset[pftype].append([ifaceR,ifaceL]) # ensure we are using the element inside the region as the left element
                feleset[etype].append(eidx)
                felespts[etype,eidx] = pts[etype][eidx,:,:]
            
            # periodic faces:
            elif (flagL & flagR) & (ifaceL[3] != 0 ): 
                pftype = self._get_pftype_from_fpairs(ifaceL,ifaceR)
                fset[pftype].append([ifaceL,ifaceR])
                fset[pftype].append([ifaceR,ifaceL])  
                
                etype, eidx = ifaceL[0:2]
                feleset[etype].append(eidx)
                felespts[etype,eidx] = pts[etype][eidx,:,:]

                etype, eidx = ifaceR[0:2]
                feleset[etype].append(eidx)
                felespts[etype,eidx] = pts[etype][eidx,:,:]

        # Collect boundary interfaces of the fwh surf and their info
        ghostR = ['ghost',-1e20,-1,-1e20]
        for k,v in mesh.items():
            if 'bcon' in k:   
                bclhs = v.astype('U4,i4,i1,i2').tolist()
                for ifaceL in bclhs:
                    etype, eidx, fidx = ifaceL[0:3]
                    if (etype in etypes) & (eidx in eset[etype]):
                        pftype = self._fnum_to_pftype(etype,fidx)
                        fset[pftype].append([ifaceL,ghostR])
                        feleset[etype].append(eidx)
                        felespts[etype,eidx] = pts[etype][eidx,:,:]

        # Collect face nodes:
        for pftype, fpairs in fset.items():
            ffp = np.moveaxis(fpairs,0,1)
            ii = 0
            for ifaceL, ifaceR in zip(ffp[0],ffp[1]):
                etype,eidx,fidx = ifaceL[:3]
                eidx = np.int(eidx)
                fidx = np.int(fidx) + self._fnum_offset[etype][pftype] 
                nelnodes = felespts[etype,eidx].shape[0]
                nidx = self._petype_fnmap[etype,nelnodes][pftype][fidx]
                nset[pftype].append(felespts[etype,eidx][nidx,:])

        self.fwhfnodes = nset
        self.fwhfset = fset
        self.fwheset = feleset
        self.fwhelespts = felespts

    def _get_pftype_from_fpairs(self,ifaceL,ifaceR):
        pftype0 = self._fnum_to_pftype(ifaceL[0],ifaceL[2])
        pftype1 = self._fnum_to_pftype(ifaceR[0],ifaceR[2])
        if pftype0 != pftype1:
            raise KeyError("pftypeL: "+f'{pftype0}'+" is not equal to pftypeR: "+f'{pftype1}')
        else:
           return pftype0

    def _fnum_to_pftype(self,inetype,fnum):
        return self._fnum_pftype_map[inetype][fnum][1]

    def _get_npts_ncells(self,mk): 
        ncells = np.asarray(self.fwhfset[mk]).shape[0]
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

    def write_vtk_out(self,fname,dsettoplogy):

        dsetname = {'unstructured': "UNSTRUCTURED_GRID", 'polydata': "POLYDATA"}

        write_s_to_fh = lambda s: fh.write(s)
        
        with open(fname, 'w', encoding="ascii") as fh:
            write_s_to_fh(f'# vtk DataFile Version {self.vtkfile_version}\n'
                          'FWH Surface\n'
                          'ASCII\n'
                          f'DATASET {dsetname[dsettoplogy]}\n')

            self._write_vtk_points(fh)
            self._write_vtk_cells(fh,dsettoplogy)

    def _write_vtk_points(self,vtkf):
        ndims=3
        fpdtype = self.fpdtype 
        vdtype = 'Float32' if fpdtype == np.float32 else 'Float64'

        write_s = lambda s: vtkf.write(s)
        npts_tot=ncells_tot=0

        for etype,v in self.fwhfnodes.items():
            npts, ncells = self._get_npts_ncells(etype)
            npts_tot += npts
            ncells_tot += ncells
        write_s(f'POINTS {npts_tot} {vdtype}\n')

        for etype,v in self.fwhfnodes.items():
            npts, ncells = self._get_npts_ncells(etype)
            vpts = np.array(self.fwhfnodes[etype]).astype(fpdtype) # eidx,nnodes,dim
            # Append dummy z dimension for points in 2D
            if self.ndims == 2:
                vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')
            vpts = vpts.reshape(-1, ndims)
            np.savetxt(vtkf,vpts,fmt='%1.16f %1.16f %1.16f')

    def _write_vtk_cells(self,vtkf,dsettoplogy):

        ncells = {}
        npts = {}
        cellsname = {'unstructured': "CELLS", 'polydata': "POLYGONS"}
        cname = cellsname[dsettoplogy]

        write_s = lambda s: vtkf.write(s)

        keys = list(self.fwhfnodes)

        ncells_tot = nsize = 0
        for etype in keys:
            npts[etype],ncells[etype] = self._get_npts_ncells(etype)
            ncells_tot += ncells[etype]

            nfopts = self._petype_focount_map[etype]
            nsize += (nfopts+1) * ncells[etype]

        write_s(f'{cname} {ncells_tot} {nsize}\n')

        # Writing cell connectivity
        npts_tot = 0
        for etype in keys:
            nfopts = self._petype_focount_map[etype]
            nodes = np.arange(nfopts)

            # Prepare VTU cell arrays
            vtk_con = np.tile(nodes, (ncells[etype], 1))
            vtk_con += (np.arange(ncells[etype])*nfopts)[:, None]
            vtk_con += npts_tot
            vtk_con = np.insert(vtk_con,0,nfopts,axis=1)
            
            np.savetxt(vtkf,vtk_con,fmt='%d')

            npts_tot += npts[etype]
        
        # Writing Cell Types:
        if cname == cellsname['unstructured']:
            write_s(f'CELL_TYPES {ncells_tot}\n')
            for etype in keys:
                # Prepare VTU cell types array
                vtk_types = np.ones(ncells[etype]) * self.vtk_types[etype]
                np.savetxt(vtkf,vtk_types,fmt='%d')
    

    def write_vtu_out(self,fname):

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
            for etype,v in self.fwhfnodes.items():
                off = self._write_serial_header(fh, etype, off)

            write_s_to_fh('</UnstructuredGrid>\n'
                          '<AppendedData encoding="raw">\n_')

            # Data
            for etype,v in self.fwhfnodes.items():
                self._write_data(fh, etype)

            write_s_to_fh('\n</AppendedData>\n</VTKFile>')

    def _write_darray(self, array, vtuf, dtype):
        array = array.astype(dtype)
        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _process_name(self, name):
        return re.sub(r'\W+', '_', name)

    def _write_serial_header(self, vtuf, mk, off):
        names, types, comps, sizes = self._get_array_attrs(mk)
        npts, ncells = self._get_npts_ncells(mk)

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


    def _write_data(self, vtuf, mk):
        fpdtype = self.fpdtype 
        vpts = np.array(self.fwhfnodes[mk]).astype(fpdtype) # eidx,nnodes,dim
        neles = self._get_npts_ncells(mk)[1]
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

    