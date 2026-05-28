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
#      script    = /path/to/pipeline.py
#      division  = 3                      ; vis subdivision (default = order)
#      clean     = true                   ; average shared-vertex values
#      volume    = true                   ; include volume source (default
#                                         ;   true when no surfaces defined)
#      surface-walls = bc/wall            ; named surface source (optional)
#      field-velocity = u, v, w           ; user-defined vector field
#      field-pressure = p                 ; user-defined scalar field
#      postproc-mach = volume             ; run mach postproc on volume
#      postproc-cf = walls                ; run cf postproc on walls surface
#
# Output cadence is controlled entirely by the Catalyst script's
# extractor triggers (TimeStep/TimeValue/Frequency/Python).  PyFR
# queries them via vtkSMExtractsController each step and only updates
# the Conduit blueprint when at least one extractor will fire.
#
# Field naming in the Conduit blueprint
# =====================================
# Every user-defined and postproc field is published namespaced by its
# source:
#
#      field-velocity = u, v, w     ->  'volume_velocity'
#      postproc-mach  = volume      ->  'volume_mach'
#      field-pressure on surface-walls -> 'walls_pressure'
#
# Each source is published as its own Catalyst channel named after the
# source ('volume', and one per surface-<name>).  The blueprint nests as:
#
#      catalyst/channels/<source>/data/domain_<N>/
#          coordsets/<source>_coords/values
#          topologies/<source>/elements
#          fields/<source>_<field>/values
#
# Note: the OFFLINE `pyfr export` writer uses unprefixed Title-Case
# names ('Velocity', 'Density').  ParaView scripts saved from such a
# .vtu must be updated to the namespaced names listed above.
#
# Creating Pipeline Scripts
# =========================
# Scripts are generated from ParaView's GUI via:
# File > Save Catalyst Script
#
# The generated script must be modified for in-situ use with PyFR:
#
#   1. Replace each file reader with a TrivialProducer whose
#      registrationName matches the source channel ('volume' or a
#      surface name).  One producer per source:
#
#        # BEFORE (ParaView-generated):
#        reader = XMLUnstructuredGridReader(
#            registrationName='volume.vtu',
#            FileName=['/path/to/volume.vtu'])
#
#        # AFTER (in-situ):
#        volume = TrivialProducer(registrationName='volume')
#        vehicle = TrivialProducer(registrationName='vehicle')
#
#   2. Update any downstream filter inputs to reference the matching
#      producer instead of the reader:
#
#        slice1 = Slice(Input=volume)    # was: Input=reader
#
#   3. Rename every field reference in the script ('Velocity',
#      'Density', etc.) to its namespaced form ('volume_velocity',
#      'volume_density', ...).  The script-side names that need
#      updating: ColorArrayName, GetColorTransferFunction(),
#      GetOpacityTransferFunction(), GetScalarBar(), and any
#      'PointData' / 'CellData' references.
#
#   4. Set each extractor's Trigger to control output cadence:
#
#        ext.Trigger = 'TimeStep'
#        ext.Trigger.Frequency = 100         # every 100 PyFR steps
#
#        ext.Trigger = 'TimeValue'
#        ext.Trigger.Length = 0.1            # every 0.1 of sim time
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

from pyfr.ctypesutil import LibWrapper, platform_libdirs, platform_libname
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.plugins.soln.insitu import (ConduitNode, ConduitWrappers,
                                      InSituError, InSituRenderer,
                                      IntegratorAdapter)


class CatalystError(InSituError): pass


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


class CatalystConduitWrappers(ConduitWrappers):
    _libname = 'catalyst'
    _functions = [(ret, f'catalyst_{fn}', *args)
                  for ret, fn, *args in ConduitWrappers._functions]

    def _load_library(self):
        return _load_catalyst_lib()

    def _transname(self, fname):
        return fname.removeprefix('catalyst_')


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


class CatalystRenderer(InSituRenderer):
    error_cls = CatalystError

    def __init__(self, adapter, isrestart):
        # External buffers must outlive each catalyst_execute call
        self._coord_bufs = []
        self._field_bufs = []

        super().__init__(adapter, isrestart)

    def _load_conduit(self):
        return CatalystConduitWrappers()

    def _init_host(self):
        comm, _, _ = get_comm_rank_root()

        self.lib = CatalystWrappers()

        local = {s for s, _ in self.dinfo}
        for sname in self.sources:
            self.mesh_n[f'catalyst/channels/{sname}/type'] = 'multimesh'
            if sname not in local:
                self.mesh_n.empty_object(f'catalyst/channels/{sname}/data')

        init_n = ConduitNode(self.conduit)
        init_n['catalyst/scripts/script0/filename'] = str(
            self.acfg.getpath(self.cfgsect, 'script', abs=True))
        init_n['catalyst/mpi_comm'] = comm.py2f()
        init_n['catalyst_load/implementation'] = 'paraview'
        self.lib.catalyst_initialize(init_n)

    def bootstrap(self, adapter):
        # Execute once to bring pipeline to life
        self.execute(adapter)
        from paraview import servermanager
        self._trigger_ctrl = servermanager.vtkSMExtractsController()

    def will_fire(self, tcurr, cycle):
        ctrl = self._trigger_ctrl
        ctrl.SetTime(float(tcurr))
        ctrl.SetTimeStep(int(cycle))
        return bool(ctrl.IsAnyTriggerActivated())

    def _domain_path(self, sname, domid):
        return f'catalyst/channels/{sname}/data/domain_{domid}'

    def _write_step_state(self, dom, tcurr, cycle):
        self.mesh_n[f'{dom}/state/time'] = float(tcurr)
        self.mesh_n[f'{dom}/state/cycle'] = cycle

    def _emit_coords(self, mesh_n, dom, cs, xyz):
        # AoS — keep buffer alive until the next catalyst_execute call
        aos = np.ascontiguousarray(np.asarray(xyz).T)
        self._coord_bufs.append(aos)
        mesh_n.set_aos(f'{dom}/coordsets/{cs}/values', 'xyz', aos)

    def _emit_field(self, mesh_n, dom, fname, arr):
        path = f'{dom}/fields/{fname}/values'
        ncomp = arr.shape[-1]

        if ncomp == 1:
            # Match Ascent's flat ordering for scalars
            mesh_n[path] = np.ascontiguousarray(arr.squeeze(-1).T)
        else:
            # AoS — direct path is (nsvpts, neles, ncomp); cleaned path
            # is (npoints, ncomp)
            if arr.ndim == 3:
                vbuf = arr.swapaxes(0, 1).reshape(-1, ncomp)
            else:
                vbuf = arr.reshape(-1, ncomp)
            vbuf = np.ascontiguousarray(vbuf)
            self._field_bufs.append(vbuf)
            mesh_n.set_aos(path, 'xyz', vbuf)

    def execute(self, adapter):
        comm, _, _ = get_comm_rank_root()

        self.mesh_n['catalyst/state/timestep'] = adapter.cycle
        self.mesh_n['catalyst/state/time'] = float(adapter.tcurr)

        # Field arrays from previous execute can now be reused/freed
        self._field_bufs.clear()

        fields = self._evaluate_exprs(adapter)
        for sname, source in self.sources.items():
            source.publish_fields(self.mesh_n, fields.get(sname, {}))

        self.lib.catalyst_execute(self.mesh_n)
        comm.barrier()

    def finalise(self):
        if lib := getattr(self, 'lib', None):
            self.lib = None
            lib.catalyst_finalize(ConduitNode(self.conduit))


class CatalystPlugin(BaseSolnPlugin):
    name = 'catalyst'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self._renderer = CatalystRenderer(
            IntegratorAdapter(intg, intg.cfg, cfgsect), intg.isrestart)
        self._bootstrap_done = False

    def __call__(self, intg):
        adapter = IntegratorAdapter(intg, intg.cfg, self.cfgsect)

        if not self._bootstrap_done:
            self._renderer.bootstrap(adapter)
            self._bootstrap_done = True
            return

        if self._renderer.will_fire(adapter.tcurr, adapter.cycle):
            self._renderer.execute(adapter)

    def finalise(self, intg):
        if r := getattr(self, '_renderer', None):
            r.finalise()
            del self._renderer
