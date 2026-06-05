from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.snapshot import IntgSnapshot
from pyfr.util import file_path_gen
from pyfr.writers.vtk.boundary import VTKBoundaryWriter
from pyfr.writers.vtk.spanwise import VTKSpanwiseWriter
from pyfr.writers.vtk.stl import VTKSTLWriter
from pyfr.writers.vtk.volume import VTKVolumeWriter


class VTUWriterPlugin(BaseSolnPlugin):
    # In-situ VTU writer.  One plugin instance = one output stream (volume
    # or one boundary).  Use the plugin suffix mechanism for multiple
    # streams: [soln-plugin-vtu-volume], [soln-plugin-vtu-walls].
    name = 'vtu'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        cfg, s = self.cfg, cfgsect

        basedir = cfg.getpath(s, 'basedir', '.', abs=True)
        basename = cfg.get(s, 'basename')
        if not basename.endswith('.vtu'):
            basename += '.vtu'
        self._fgen = file_path_gen(basedir, basename, intg.isrestart)

        # Output cadence
        self.dt_out = cfg.getfloat(s, 'dt-out')
        self.tout_last = intg.tcurr
        if not intg.isrestart:
            self.tout_last -= self.dt_out
        intg.call_plugin_dt(intg.tcurr, self.dt_out)

        # Writer options
        order = cfg.getint(s, 'order', 0) or None
        divisor = cfg.getint(s, 'divisor', 0) or None
        prec = cfg.get(s, 'precision', 'single')
        clean = cfg.getbool(s, 'clean', True)

        adds = cfg.get(s, 'add-fields', '').split(',')
        removes = cfg.get(s, 'remove-fields', '').split(',')
        add_fields = [n.strip() for n in adds if n.strip()]
        remove_fields = [n.strip() for n in removes if n.strip()]

        mode = cfg.get(s, 'mode', 'volume').strip()
        common = dict(prec=prec, order=order, divisor=divisor,
                      add_fields=add_fields, remove_fields=remove_fields,
                      discontinuous=not clean)
        mesh, scfg = intg.system.mesh, intg.cfg

        if mode == 'volume':
            self._writer = VTKVolumeWriter(mesh, scfg, **common)
        elif mode == 'boundary':
            bnd = cfg.get(s, 'boundary', '').strip()
            self._writer = VTKBoundaryWriter(mesh, scfg,
                                             boundaries=[bnd] if bnd else [],
                                             **common)
        elif mode == 'spanwise':
            periodic = cfg.get(s, 'periodic', '').strip() or None
            sw_boundary = cfg.get(s, 'boundary', '').strip() or None
            nstations = cfg.getint(s, 'nstations', 0) or None
            self._writer = VTKSpanwiseWriter(mesh, scfg, periodic=periodic,
                                             boundary=sw_boundary,
                                             nstations=nstations, **common)
        elif mode == 'stl':
            stl_raw = cfg.get(s, 'stl', '').strip()
            stl_regions = [r.strip() for r in stl_raw.split(',') if r.strip()]
            subdiv = cfg.get(s, 'subdiv', 'linear')
            self._writer = VTKSTLWriter(mesh, scfg, stl_regions, subdiv=subdiv,
                                        **common)
        else:
            raise ValueError(f'{s}: unknown mode={mode!r}; choose volume, '
                             'boundary, spanwise, or stl')

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        outfname = str(self._fgen.send(intg.tcurr))
        self._writer.emit(IntgSnapshot(intg), outfname)
        self.tout_last = intg.tcurr
