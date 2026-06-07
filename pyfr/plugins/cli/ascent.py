from pyfr.inifile import Inifile
from pyfr.mpiutil import init_mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.plugins.soln.ascent import AscentRenderer
from pyfr.snapshot import from_file as snap_from_file


class AscentCLIPlugin(BaseCLIPlugin):
    name = 'ascent'

    @classmethod
    def add_cli(cls, parser):
        sp = parser.add_subparsers()

        # Render command
        ap_render = sp.add_parser('render', help='ascent render --help')
        ap_render.set_defaults(process=cls.render_cli)
        ap_render.add_argument('mesh', help='mesh file')
        ap_render.add_argument('solns', nargs='*', help='solution files')
        ap_render.add_argument('cfg', help='ascent config file')
        ap_render.add_argument('--cfgsect', help='ascent config file section')

    @cli_external
    def render_cli(self, args):
        # Initialise MPI
        init_mpi()

        acfg = Inifile.load(args.cfg)
        acfgsect = args.cfgsect or acfg.sections()[0]

        # Current Ascent render and associated config
        renderer, rcfg = None, None

        # Iterate over the solutions
        for s in args.solns:
            # Open the solution and create an Ascent adapter
            snap = snap_from_file(args.mesh, s)

            # See if we need to create a new Ascent renderer
            if not renderer or rcfg != snap.cfg:
                renderer = AscentRenderer(snap.mesh, snap.cfg, acfgsect,
                                          isrestart=True, acfg=acfg)
                rcfg = snap.cfg

            # Perform the rendering
            renderer.render(snap)
