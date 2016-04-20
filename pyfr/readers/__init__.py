# -*- coding: utf-8 -*-

from pyfr.readers.base import BaseReader, NodalMeshAssembler
from pyfr.readers.cgns import CGNSReader
from pyfr.readers.gmsh import GmshReader

from pyfr.util import subclasses, subclass_where


def get_reader_by_name(name, *args, **kwargs):
    return subclass_where(BaseReader, name=name)(*args, **kwargs)


def get_reader_by_extn(extn, *args, **kwargs):
    reader_map = {ex: cls
                  for cls in subclasses(BaseReader)
                  for ex in cls.extn}

    return reader_map[extn](*args, **kwargs)
