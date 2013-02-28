# -*- coding: utf-8 -*-

from pyfr.readers.base import BaseReader
from pyfr.readers.gmsh import GmshReader

from pyfr.util import all_subclasses, subclass_map


def get_reader_by_name(name, *args, **kwargs):
    reader_map = subclass_map(BaseReader, 'name')

    return reader_map[name](*args, **kwargs)


def get_reader_by_extn(extn, *args, **kwargs):
    reader_map = {ex: cls
                  for cls in all_subclasses(BaseReader)
                  for ex in cls.extn}

    return reader_map[extn](*args, **kwargs)
