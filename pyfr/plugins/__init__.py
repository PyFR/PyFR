from pyfr.plugins.base import BasePlugin, BaseCLIPlugin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.plugins import cli, postproc, soln, solver
from pyfr.util import subclass_where


def get_plugin(prefix, name, *args, **kwargs):
    cls = subclass_where(BasePlugin, prefix=prefix, name=name)
    return cls(*args, **kwargs)
