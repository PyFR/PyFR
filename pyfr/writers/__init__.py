from pyfr.util import subclasses, subclass_where
from pyfr.writers.base import BaseWriter
from pyfr.writers.vtk import (VTKBoundaryWriter, VTKSpanwiseWriter,
                              VTKSTLWriter, VTKVolumeWriter)


def writer_cls_by_name(name, type):
    return subclass_where(BaseWriter, name=name, type=type)


def writer_cls_by_extn(extn, type):
    return next(cls for cls in subclasses(BaseWriter, just_leaf=True)
                if cls.type == type and extn in cls.extn)
