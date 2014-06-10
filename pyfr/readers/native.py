# -*- coding: utf-8 -*-

"""Allows for interopability between .pyfr{m, s}-{file, dir} archive formats

"""
from collections import Mapping, OrderedDict
import errno
import os

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclasses, lazyprop


class PyFRBaseReader(Mapping):
    """Contains frequently used operations for PyFR archive formats"""

    def list_archives_startswith(self, prefix):
        """Lists the file names in an archive object that begin with prefix

        :param prefix: Prefix of file names to be listed.
        :type prefix: string

        :return: File names beginning with prefix.
        :rtype: list

        """
        return [name for name in self if name.startswith(prefix)]

    @property
    def spt_files(self):
        """Lists the names of files in the archive that contain mesh data

        :return: File names beginning with 'spt'.
        :rtype: list

        """
        return self.list_archives_startswith('spt')

    @property
    def soln_files(self):
        """Lists the names of files in the archive that contain solution data

        :return: List of file names beginning with 'soln'.
        :rtype: list

        """
        return self.list_archives_startswith('soln')

    @lazyprop
    def array_info(self):
        """Indexes element types and data array shapes in an archive by file

        Returns a dictionary of information about the numpy array
        files containing pyfr {mesh, solution} data in a .pyfr{m, s}
        archive.

        The dictionary is indexed by array file name, and is ordered
        such that element type counts faster than partition number.
        Dictionary values contain the pyfr element type as a string,
        and the shape of the array as a tuple.

        :return: Element type and array shape of data arrays, indexed
                 by array file name.
        :rtype: :class:`collections.OrderedDict`

        :Example:

        >>> from pyfr.readers import native
        >>> example = read_pyfr_data('example.pyfrm')
        >>> example.get_array_info
        OrderedDict([('spt_hex_p0', ('hex', (27,  4, 3))),
                     ('spt_tet_p0', ('tet', (27, 20, 3))),
                     ('spt_hex_p1', ('hex', (27,  8, 3)))])

        """
        info = OrderedDict()

        # Retrieve list of {mesh, solution} array names, and set name prefix.
        if self.spt_files:
            ls_files = self.spt_files
            prfx = 'spt'

        elif self.soln_files:
            ls_files = self.soln_files
            prfx = 'soln'

        else:
            raise RuntimeError('"%s" does not contain solution or shape point '
                               'files' % (self.fname))

        # Element types known to PyFR
        eletypes = [b.name for b in subclasses(BaseShape)
                    if hasattr(b, 'name')]

        # Assembles possible array file names, then checks if present
        # in .pyfr{m, s} archive.  If so, the element type and array
        # shape are assigned to the array file name in info.
        prt = 0
        while len(info) < len(ls_files):
            for et in eletypes:
                name = '%s_%s_p%d' % (prfx, et, prt)

                if name in ls_files:
                    info[name] = (et, self[name].shape)

            prt += 1

        return info


class PyFRDirReader(PyFRBaseReader):
    """Contains requisite python magic methods for .pyfr{m, s}-dir archives"""
    def __init__(self, fname):
        self.fname = fname

    def __getitem__(self, aname):
        try:
            return np.load(os.path.join(self.fname, aname + '.npy'),
                           mmap_mode='r')
        except IOError as e:
            if e.errno == errno.ENOENT:
                raise KeyError
            else:
                raise

    def __iter__(self):
        return (name.rsplit('.')[0] for name in os.listdir(self.fname))

    def __len__(self):
        return len(os.listdir(self.fname))


class PyFRFileReader(PyFRBaseReader):
    """Contains requisite python magic methods for .pyfr{m, s}-file archives"""
    def __init__(self, fname):
        self._npf = np.load(fname)

    def __getitem__(self, aname):
        return self._npf[aname]

    def __iter__(self):
        return iter(self._npf.files)

    def __len__(self):
        return len(self._npf.files)


def read_pyfr_data(fname):
    """Reads .pyfr{m, s}-{file, dir} archive

    The .pyfr{m, s}-{file, dir} archive is checked for being of type
    directory. The name is then passed to initialize an object of
    either :py:class:`~pyfr.readers.native.PyFRDirReader` or
    :py:class:`~pyfr.readers.native.PyFRFileReader` class.

    :param fname: File name of .pyfr{m, s}-{file, dir} archive to be read.
    :type fname: string
    :rtype: :class:`~pyfr.readers.native.PyFRDirReader`,
            :class:`~pyfr.readers.native.PyFRFileReader`

    :Example:

    >>> from pyfr.readers import native
    >>> example = read_pyfr_data('example.pyfrs')
    >>> list(example)
    ['config', 'soln_hex_p0', 'soln_hex_p1', 'mesh_uuid', 'stats']
    >>> len(example)
    5
    >>> example['mesh_uuid']
    array('484cde08-a27d-43c8-a8e6-3a5bc49c3326', dtype='|S36')
    >>> example.soln_files
    ['soln_hex_p0', 'soln_hex_p1']
    >>> example.spt_files
    []
    >>> example.array_info
    OrderedDict([('soln_hex_p0', ('hex', (64, 4, 5))),
                 ('soln_hex_p1', ('hex', (64, 4, 5)))])

    .. note::
        The pyfr{m, s}-dir format offers superior memory performance
        for most numpy intrinsic functions.  :class:`numpy.memmap` mode
        is leveraged, which allows numpy arrays to be accessed without
        being read into memory unnecessarily.

    """
    if os.path.isdir(fname):
        return PyFRDirReader(fname)
    else:
        return PyFRFileReader(fname)
