from pyfr.plugins.postproc.cf import CfPostProc
from pyfr.plugins.postproc.cp import CpPostProc
from pyfr.plugins.postproc.isen_mach import IsentropicMachPostProc
from pyfr.plugins.postproc.mach import MachPostProc
from pyfr.plugins.postproc.vorticity import VorticityPostProc
from pyfr.plugins.postproc.yplus import YPlusPostProc
from pyfr.plugins.base import BasePlugin
from pyfr.util import subclass_where


def get_postproc_plugin(name, *args, **kwargs):
    cls = subclass_where(BasePlugin, prefix='postproc', name=name)
    return cls(*args, **kwargs)
