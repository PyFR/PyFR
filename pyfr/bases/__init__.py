# -*- coding: utf-8 -*-

from pyfr.bases.base import BaseBasis
from pyfr.bases.mixed import PriBasis
from pyfr.bases.tensorprod import HexBasis, QuadBasis
from pyfr.bases.simplex import TetBasis, TriBasis
from pyfr.util import subclass_map


def get_std_ele_by_name(name, order):
    """ Gets the shape points of a standard element of name and order

    :param name: Name of PyFR element type
    :param order: Shape order of element
    :type name: string
    :type order: integer
    :rtype: np.ndarray

    """
    ele_cls = subclass_map(BaseBasis, 'name')[name]

    return ele_cls.std_ele(order)
