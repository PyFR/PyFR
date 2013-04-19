# -*- coding: utf-8 -*-

from pyfr.writers.base import BaseWriter
from pyfr.writers.paraview import ParaviewWriter

from pyfr.util import all_subclasses, subclass_map


def get_writer_by_name(name, *args, **kwargs):
    writer_map = subclass_map(BaseWriter, 'name')

    return writer_map[name](*args, **kwargs)


def get_writer_by_extn(extn, *args, **kwargs):
    writer_map = {ex: cls
                  for cls in all_subclasses(BaseWriter)
                  for ex in cls.extn}

    return writer_map[extn](*args, **kwargs)
