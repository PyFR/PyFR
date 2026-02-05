from pyfr.util import subclasses, subclass_where
from pyfr.writers.base import BaseWriter
from pyfr.writers.vtk import VTKBoundaryWriter, VTKSTLWriter, VTKVolumeWriter


def get_writer_by_name(name, type, /, *kargs, **kwargs):
    return subclass_where(BaseWriter, name=name, type=type)(*kargs, **kwargs)


def get_writer_by_extn(extn, type, /, *kargs, **kwargs):
    writer_map = {ex: cls
                  for cls in subclasses(BaseWriter, just_leaf=True)
                  for ex in cls.extn
                  if cls.type == type}

    return writer_map[extn](*kargs, **kwargs)
