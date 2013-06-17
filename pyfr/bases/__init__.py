# -*- coding: utf-8 -*-

from pyfr.bases.base import BasisBase
from pyfr.bases.tensorprod import (get_std_hex, get_std_quad, HexBasis,
                                   QuadBasis)


def get_std_ele_by_name(name, order):
    """ Gets the shape points of a standard element of name and order

    :param name: Name of PyFR element type
    :param order: Shape order of element
    :type name: string
    :type order: integer
    :rtype: np.ndarray

    """
    ele_map = {'quad': get_std_quad, 'hex': get_std_hex}

    return ele_map[name](order)
