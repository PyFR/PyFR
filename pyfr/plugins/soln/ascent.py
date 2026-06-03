from ctypes import c_void_p

from pyfr.ctypesutil import LibWrapper
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.plugins.soln.insitu import (ConduitNode, InSituError,
                                      InSituRenderer)
from pyfr.snapshot import IntgSnapshot
from pyfr.util import file_path_gen, first


class AscentError(InSituError): pass


def bp_key(k):
    return k.replace('_', '/').replace('-', '_')


class AscentWrappers(LibWrapper):
    _libname = 'ascent_mpi'

    # Functions
    _functions = [
        (None, 'ascent_close', c_void_p),
        (c_void_p, 'ascent_create', c_void_p),
        (None, 'ascent_execute', c_void_p, c_void_p),
        (None, 'ascent_open', c_void_p, c_void_p),
        (None, 'ascent_publish', c_void_p, c_void_p)
    ]


class AscentRenderer(InSituRenderer):
    error_cls = AscentError

    def __init__(self, mesh, scfg, cfgsect, isrestart, *, acfg=None):
        super().__init__(mesh, scfg, cfgsect, isrestart, acfg=acfg)

    def __del__(self):
        if getattr(self, 'ascent_ptr', None) and getattr(self, 'lib', None):
            self.lib.ascent_close(self.ascent_ptr)
            self.ascent_ptr = None

    def _init_host_publish(self):
        self.basedir = self.acfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        self._image_paths = []
        self._init_scenes()
        self._init_pipelines()

    def _init_host(self):
        comm, _, _ = get_comm_rank_root()

        self.lib = lib = AscentWrappers()
        self.ascent_ptr = lib.ascent_create(None)

        self.ascent_config = ConduitNode(self.conduit)
        self.ascent_config['mpi_comm'] = comm.py2f()
        self.ascent_config['runtime/type'] = 'ascent'
        backend = self.acfg.get(self.cfgsect, 'viskores-backend', 'serial')
        self.ascent_config['runtime/viskores/backend'] = backend

        # Disable autoload of ascent_actions.{yaml,json}
        self.ascent_config['actions_file'] = ''

        # Open an Ascent instance
        lib.ascent_open(self.ascent_ptr, self.ascent_config)

        # Pre configure scenes and pipelines
        self.actions = ConduitNode(self.conduit)
        self._add_scene = self.actions.append()
        self._add_scene['action'] = 'add_scenes'
        self._add_scene['scenes'] = self.scenes

        self._add_pipeline = self.actions.append()
        self._add_pipeline['action'] = 'add_pipelines'
        self._add_pipeline['pipelines'] = self.pipelines

    def _init_pipelines(self):
        self.pipelines = pl = ConduitNode(self.conduit)

        for k in self.acfg.items(self.cfgsect, prefix='pipeline-'):
            pn = k.removeprefix('pipeline-')
            cfg = self.acfg.getliteral(self.cfgsect, k)
            cfg = [cfg] if isinstance(cfg, dict) else cfg

            for j, filt in enumerate(cfg):
                params = ConduitNode(self.conduit)

                pl[f'pl_{pn}/f{j}/type'] = filt.pop('type')
                for kf, vf in filt.items():
                    if kf == 'output-name':
                        if vf in self._fields_write:
                            raise KeyError(f'Output name {vf!r} already used')
                        self._fields_write.add(vf)
                    elif kf == 'field':
                        self._fields_read.add(vf)

                    params[bp_key(kf)] = vf

                pl[f'pl_{pn}/f{j}/params'] = params

    def _init_scenes(self):
        self.scenes = ConduitNode(self.conduit)

        for k in self.acfg.items(self.cfgsect, prefix='scene-'):
            sn = k.removeprefix('scene-')
            cfg = self.acfg.getliteral(self.cfgsect, k)

            plots = cfg.get('plots')
            if not isinstance(plots, list) or not plots:
                raise AscentError(f'Scene {sn!r} must define plots = [...]')

            for j, plot in enumerate(plots):
                self._init_plot(sn, f'p{j}', plot)

            for kc, vc in cfg.items():
                if kc.startswith('render-'):
                    rname = kc.removeprefix('render-')
                    self._render_options(f's_{sn}/renders/r_{rname}', vc)

            # Synthesise a single default render from scene-level keys
            keys = ('image-name', 'image-prefix')
            defaults = {k: cfg[k] for k in keys if k in cfg}
            has_render = any(kc.startswith('render-') for kc in cfg)
            if defaults and not has_render:
                self._render_options(f's_{sn}/renders/r_default', defaults)
            elif not has_render:
                raise KeyError(f'No render config given for scene {sn!r}')

    def _init_plot(self, sn, pname, plot):
        # Default to the only source when there is no ambiguity
        if (src := plot.get('source')) is None:
            if len(self.regions) > 1:
                raise AscentError(f'Plot {pname!r} of scene {sn!r} needs '
                                  'source= when multiple sources exist')
            src = first(self.regions)

        # Plot field references are auto-namespaced to the plot's source
        if (field := plot.get('field')) is not None:
            field = self._field_name(src, field)
            self._fields_read.add(field)

        for kc, vc in plot.items():
            if kc.startswith('render-') or kc == 'source':
                continue
            if kc == 'pipeline':
                self.scenes[f's_{sn}/plots/{pname}/pipeline'] = f'pl_{vc}'
            elif kc == 'field':
                self.scenes[f's_{sn}/plots/{pname}/field'] = field
            else:
                self.scenes[f's_{sn}/plots/{pname}/{bp_key(kc)}'] = vc

    def _render_options(self, path, opts):
        for k, v in opts.items():
            if k in ('image-name', 'image-prefix'):
                continue

            self.scenes[f'{path}/{bp_key(k)}'] = v

        if (name := opts.get('image-name')) is not None:
            gen = file_path_gen(self.basedir, name, self.isrestart,
                                extn='.png')
            self._image_paths.append((f'scenes/{path}/image_name', gen))
        elif (prefix := opts.get('image-prefix')) is not None:
            self.scenes[f'{path}/image_prefix'] = f'{self.basedir}/{prefix}'
        else:
            raise KeyError(f'Render at {path!r} needs image-name or '
                           'image-prefix')

    def render(self, snap):
        comm, _, _ = get_comm_rank_root()

        # Set file names from the per-call snap's tcurr
        for path, gen in self._image_paths:
            self._add_scene[path] = str(gen.send(snap.tcurr))

        fields = self._evaluate_exprs(snap)
        self.publish(fields)

        self.lib.ascent_publish(self.ascent_ptr, self.mesh_n)
        self.lib.ascent_execute(self.ascent_ptr, self.actions)

        comm.barrier()

    def finalise(self):
        if lib := getattr(self, 'lib', None):
            self.lib = None
            lib.ascent_close(self.ascent_ptr)


class AscentPlugin(BaseSolnPlugin):
    name = 'ascent'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self._renderer = AscentRenderer(intg.system.mesh, intg.cfg, cfgsect,
                                        intg.isrestart)

    def __call__(self, intg):
        self._renderer.render(IntgSnapshot(intg))

    def finalise(self, intg):
        if r := getattr(self, '_renderer', None):
            r.finalise()
            del self._renderer
