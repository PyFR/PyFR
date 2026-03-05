# ParaView Catalyst 2.0 In-Situ Visualization Plugin
#
# Prerequisites
# =============
# Both Catalyst and ParaView must be installed into the same prefix
# ($PV_PREFIX below).  ParaView must be built against the same Python
# interpreter and mpi4py / MPI installation used by PyFR.
#
# 1. Build and install the Catalyst stub library:
#
#      git clone https://gitlab.kitware.com/paraview/catalyst.git
#      cmake -S catalyst -B catalyst-build \
#            -DCMAKE_INSTALL_PREFIX=$PV_PREFIX \
#            -DCATALYST_USE_MPI=ON \
#            -DCATALYST_WRAP_PYTHON=ON
#      cmake --build catalyst-build && cmake --install catalyst-build
#
# 2. Build and install ParaView into the SAME prefix:
#
#      cmake -S ParaView-v6.x -B paraview-build \
#            -DCMAKE_INSTALL_PREFIX=$PV_PREFIX \
#            -DCMAKE_PREFIX_PATH=$PV_PREFIX \
#            -DPARAVIEW_BUILD_EDITION=CATALYST_RENDERING \
#            -DPARAVIEW_ENABLE_CATALYST=ON \
#            -DPARAVIEW_USE_MPI=ON \
#            -DPARAVIEW_USE_PYTHON=ON \
#            -DPython3_EXECUTABLE=$(which python3) \
#            -DVTK_MODULE_USE_EXTERNAL_VTK_mpi4py=ON
#      cmake --build paraview-build && cmake --install paraview-build
#
#    Build editions (PARAVIEW_BUILD_EDITION):
#      CATALYST           — data extracts only (VTP, CSV, etc.)
#      CATALYST_RENDERING — adds headless rendering (PNG image export)
#
#    Note: PARAVIEW_ENABLE_CATALYST is OFF by default even when using
#    a CATALYST build edition — it must be explicitly enabled.
#
# Running
# =======
#      PYFR_CATALYST_LIBRARY_PATH=$PV_PREFIX/lib/libcatalyst.dylib \
#          pyfr run -b openmp mesh.pyfrm input.ini
#
# INI Configuration
# =================
#      [soln-plugin-catalyst]
#      dt-out = 1e-4
#      script = /path/to/pipeline.py
#
# Creating Pipeline Scripts
# =========================
# Scripts are generated from ParaView's GUI via:
# File > Save Catalyst Script
#
# The generated script must be modified for in-situ use with PyFR:
#
#   1. Replace the file reader with a TrivialProducer.  The
#      registrationName must be 'mesh' to match the channel name
#      used by this plugin:
#
#        # BEFORE (ParaView-generated):
#        reader = XMLUnstructuredGridReader(
#            registrationName='result.vtu',
#            FileName=['/path/to/result.vtu'])
#
#        # AFTER (in-situ):
#        mesh = TrivialProducer(registrationName='mesh')
#
#   2. Update any downstream filter inputs to reference the new
#      TrivialProducer instead of the reader:
#
#        slice1 = Slice(Input=mesh)    # was: Input=reader
#
# VTK MPI Patches
# ===============
# VTK (as of 6.1.0-RC1) has unguarded MPI_Comm_free calls that fire
# after MPI_Finalize, causing an abort at shutdown.  Two source patches
# are required to remove miss-timed MPI_Finalize error messages:
#
# 1. VTK/Parallel/MPI/vtkMPICommunicator.cxx — destructor (~line 646):
#
#      if (this->MPIComm)
#      {
#    +   int mpi_finalized = 0;
#    +   MPI_Finalized(&mpi_finalized);
#    +   if (!mpi_finalized && this->MPIComm->Handle && !this->KeepHandle)
#    -   if (this->MPIComm->Handle && !this->KeepHandle)
#
# 2. ThirdParty/IceT/vtkicet/src/communication/mpi.c — MPIDestroy:
#
#      static void MPIDestroy(IceTCommunicator self)
#      {
#    +     int mpi_finalized = 0;
#    +     MPI_Finalized(&mpi_finalized);
#    +     if (!mpi_finalized) {
#              MPI_Comm_free((MPI_Comm *)self->data);
#    +     }
#
# Rebuild and reinstall ParaView after applying these patches.

import ctypes
from ctypes import RTLD_GLOBAL, c_int, c_void_p
import os
from pathlib import Path

import numpy as np

from pyfr.conduit import ConduitError, ConduitNode, ConduitWrappers
from pyfr.ctypesutil import LibWrapper, platform_libdirs, platform_libname
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.common import region_data
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.shapes import get_vtk_shape


class CatalystError(Exception): pass


def _load_catalyst_lib():
    lpath = os.environ.get('PYFR_CATALYST_LIBRARY_PATH')
    if lpath:
        return ctypes.PyDLL(lpath, mode=RTLD_GLOBAL)

    lname = platform_libname('catalyst')
    for sd in platform_libdirs():
        try:
            return ctypes.PyDLL(str(Path(sd, lname).absolute()),
                                mode=RTLD_GLOBAL)
        except OSError:
            pass

    return ctypes.PyDLL(lname, mode=RTLD_GLOBAL)


class CatalystConduitWrappers(LibWrapper):
    _libname = 'catalyst'
    _errtype = c_void_p
    _mode = RTLD_GLOBAL
    _functions = [(ret, f'catalyst_{fn}', *args)
                  for ret, fn, *args in ConduitWrappers._functions]

    def _load_library(self):
        return _load_catalyst_lib()

    def _transname(self, fname):
        return fname.removeprefix('catalyst_')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.conduit_datatype_sizeof_index_t() != 8:
            raise RuntimeError('Conduit must be compiled with 64-bit index '
                               'types')

    def _errcheck(self, status, fn, args):
        if not status:
            raise ConduitError

        return status


def _load_conduit():
    try:
        return ConduitWrappers()
    except OSError:
        return CatalystConduitWrappers()


class CatalystWrappers(LibWrapper):
    _libname = 'catalyst'
    _mode = RTLD_GLOBAL

    _functions = [
        (c_int, 'catalyst_initialize', c_void_p),
        (c_int, 'catalyst_execute', c_void_p),
        (c_int, 'catalyst_finalize', c_void_p),
    ]

    def _errcheck(self, status, fn, _):
        if status:
            raise CatalystError(f'{fn.__name__} returned error {status}')
        return status

    def _load_library(self):
        return _load_catalyst_lib()


class _CatalystProcessor:
    bp_emap = {'hex': 'hex', 'pri': 'wedge', 'pyr': 'pyramid', 'quad': 'quad',
               'tet': 'tet', 'tri': 'tri'}

    def __init__(self, intg, cfgsect):
        comm, rank, root = get_comm_rank_root()

        cfg = intg.cfg
        system = intg.system
        sorder = cfg.getint('solver', 'order')
        divisor = cfg.getint(cfgsect, 'division', sorder)

        self.conduit = _load_conduit()
        self.lib = CatalystWrappers()

        self._elementscls = system.elementscls
        self._pnames = system.elementscls.privars(system.ndims, cfg)
        self._vvars = system.elementscls.visvars(system.ndims, cfg)
        self._scfg = cfg

        self.exec_n = ConduitNode(self.conduit)
        self.exec_n['catalyst/channels/mesh/type'] = 'multimesh'

        rdata = region_data(cfg, cfgsect, system.mesh)
        doff = comm.exscan(len(rdata)) or 0

        self._ele_regions = []
        self._coord_bufs = []
        for i, (etype, eidxs) in enumerate(rdata.items()):
            self._build_blueprint(intg, doff + i, etype, eidxs,
                                  divisor)

        init_n = ConduitNode(self.conduit)
        init_n['catalyst/scripts/script0/filename'] = str(
            cfg.getpath(cfgsect, 'script', abs=True))
        init_n['catalyst/mpi_comm'] = comm.py2f()
        init_n['catalyst_load/implementation'] = 'paraview'
        self.lib.catalyst_initialize(init_n)

    def finalise(self):
        if lib := getattr(self, 'lib', None):
            self.lib = None
            lib.catalyst_finalize(ConduitNode(self.conduit))

    def __del__(self):
        self.finalise()

    def _build_blueprint(self, intg, domid, etype, rgn, divisor):
        exec_n = self.exec_n
        pfx = f'catalyst/channels/mesh/data/domain_{domid}'
        e_str = f'{pfx}/topologies/mesh/elements'
        system = intg.system

        eles = system.ele_map[etype]
        shapecls = subclass_where(BaseShape, name=etype)
        shape = shapecls(eles.nspts, intg.cfg)

        svpts = shape.std_ele(divisor)
        soln_op = shape.ubasis.nodal_basis_at(svpts).astype(
            system.backend.fpdtype)
        xd = eles.ploc_at_np(svpts)

        self._ele_regions.append(
            (pfx, system.ele_types.index(etype), rgn, soln_op))

        xd = xd[..., rgn].transpose(1, 2, 0)
        ndims_d, neles, nsvpts = xd.shape

        exec_n[f'{pfx}/state/domain_id'] = domid
        exec_n[f'{pfx}/coordsets/coords/type'] = 'explicit'
        exec_n[f'{pfx}/topologies/mesh/coordset'] = 'coords'
        exec_n[f'{pfx}/topologies/mesh/type'] = 'unstructured'

        xd_aos = np.ascontiguousarray(xd.reshape(ndims_d, -1).T)
        self._coord_bufs.append(xd_aos)
        exec_n.set_aos(f'{pfx}/coordsets/coords/values', 'xyz', xd_aos)

        subdiv = get_vtk_shape(etype, divisor)
        snodes = subdiv.subnodes

        sconn = np.tile(snodes, (neles, 1))
        sconn += (np.arange(neles) * nsvpts)[:, None]
        exec_n[f'{e_str}/connectivity'] = sconn

        if len(scells := set(subdiv.subcells)) > 1:
            exec_n[f'{e_str}/shape'] = 'mixed'

            for sc in scells:
                an = self.bp_emap[sc]
                exec_n[f'{e_str}/shape_map/{an}'] = subdiv.vtk_types[sc]

            exec_n[f'{e_str}/shapes'] = np.tile(subdiv.subcelltypes, neles)

            scell_s = [subdiv.vtk_nodes[sc] for sc in subdiv.subcells]
            exec_n[f'{e_str}/sizes'] = np.tile(scell_s, neles)

            scell_o = np.tile(subdiv.subcelloffs, (neles, 1))
            scell_o += (np.arange(neles) * len(snodes))[:, None]
            scell_o = np.concatenate(([0], scell_o.flat[:-1]))
            exec_n[f'{e_str}/offsets'] = scell_o
        else:
            exec_n[f'{e_str}/shape'] = self.bp_emap[etype]

        for vname in self._vvars:
            fname = vname.title()
            exec_n[f'{pfx}/fields/{fname}/association'] = 'vertex'
            exec_n[f'{pfx}/fields/{fname}/volume_dependent'] = 0
            exec_n[f'{pfx}/fields/{fname}/topology'] = 'mesh'

    def execute(self, intg):
        comm = get_comm_rank_root()[0]

        exec_n = self.exec_n
        soln = intg.soln
        elementscls = self._elementscls

        exec_n['catalyst/state/timestep'] = intg.nacptsteps
        exec_n['catalyst/state/time'] = float(intg.tcurr)

        # Keep AoS field arrays alive across all element types until
        # after catalyst_execute (set_aos uses external pointers)
        field_bufs = []

        for pfx, eidx, rgn, soln_op in self._ele_regions:
            exec_n[f'{pfx}/state/time'] = intg.tcurr
            exec_n[f'{pfx}/state/cycle'] = intg.nacptsteps

            csolns = soln[eidx][..., rgn].swapaxes(0, 1)
            csolns = soln_op @ csolns

            psolns = elementscls.con_to_pri(csolns, self._scfg)
            psolns_d = dict(zip(self._pnames, psolns))

            for vname, vcomps in self._vvars.items():
                fname = vname.title()
                fpath = f'{pfx}/fields/{fname}/values'
                if len(vcomps) == 1:
                    exec_n[fpath] = psolns_d[vcomps[0]].T
                else:
                    vbuf = np.ascontiguousarray(
                        np.stack([psolns_d[c].T for c in vcomps],
                                 axis=-1).reshape(-1, len(vcomps)))
                    field_bufs.append(vbuf)
                    exec_n.set_aos(fpath, 'xyz', vbuf)

        self.lib.catalyst_execute(exec_n)
        comm.barrier()


class CatalystPlugin(BaseSolnPlugin):
    name = 'catalyst'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        self._processor = _CatalystProcessor(intg, cfgsect)

        intg.call_plugin_dt(intg.tcurr, self.dt_out)

        if not intg.isrestart:
            self.tout_last -= self.dt_out

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        self._processor.execute(intg)
        self.tout_last = intg.tcurr

    def finalise(self, intg):
        self._processor.finalise()
        del self._processor
