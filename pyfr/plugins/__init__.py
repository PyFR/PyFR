from pyfr.plugins.base import BasePlugin, BaseCLIPlugin
from pyfr.util import subclass_where


def get_plugin(prefix, name, *args, **kwargs):
    from pyfr.plugins import soln, solver
    cls = subclass_where(BasePlugin, prefix=prefix, name=name)
    return cls(*args, **kwargs)
