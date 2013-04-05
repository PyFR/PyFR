# -*- coding: utf-8 -*-

"""Mappings between the ordering of PyFR nodes, and those of external formats

"""
import numpy as np


class GmshNodeMaps(object):
    """Mappings between the node ordering of PyFR and that of Gmsh

    Node mappings are contained within two dictionaries; one maps from
    Gmsh node ordering to PyFR, and the other provides the inverse.
    Dictionary items are keyed by a tuple of element type (string) and
    number of solution points per element (integer).

    Each dictionary value is a list of integers that provide mappings
    via their list index. When lists in the "gmsh_to_pyfr" dictionary
    are indexed using the Gmsh node number, they return the equivalent
    PyFR node number.  The reverse is true for the "pyfr_to_gmsh"
    dictionary.

    :Example: Convert Gmsh node number 4, in a 64 point hexahedra, to
              the equivalent node number in PyFR:

    >>> from pyfr.readers.nodemaps import GmshNodeMaps
    >>> GmshNodeMaps.gmsh_to_pyfr['hex', 64][4]
    >>> 48
    """
    gmsh_to_pyfr = {
        ('hex', 8): np.array([0, 1, 3, 2, 4, 5, 7, 6]),
        ('hex', 27): np.array([0, 2, 8, 6, 18, 20, 26, 24, 1, 3, 9, 5, 11, 7,
                               17, 15, 19, 21, 23, 25, 4, 10, 12, 14, 16, 22,
                               13]),
        ('hex', 64): np.array([0, 3, 15, 12, 48, 51, 63, 60, 1, 2, 4, 8, 16,
                               32, 7, 11, 19, 35, 14, 13, 31, 47, 28, 44, 49,
                               50, 52, 56, 55, 59, 62, 61, 5, 9, 10, 6, 17, 18,
                               34, 33, 20, 36, 40, 24, 23, 27, 43, 39, 30, 29,
                               45, 46, 53, 54, 58, 57, 21, 22, 26, 25, 37, 38,
                               42, 41]),
        ('hex', 125): np.array([0, 4, 24, 20, 100, 104, 124, 120, 1, 2, 3, 5,
                                10, 15, 25, 50, 75, 9, 14, 19, 29, 54, 79, 23,
                                22, 21, 49, 74, 99, 45, 70, 95, 101, 102, 103,
                                105, 110, 115, 109, 114, 119, 123, 122, 121, 6,
                                16, 18, 8, 11, 17, 13, 7, 12, 26, 28, 78, 76,
                                27, 53, 77, 51, 52, 30, 80, 90, 40, 55, 85, 65,
                                35, 60, 34, 44, 94, 84, 39, 69, 89, 59, 64, 48,
                                46, 96, 98, 47, 71, 97, 73, 72, 106, 108, 118,
                                116, 107, 113, 117, 111, 112, 31, 33, 43, 41,
                                81, 83, 93, 91, 32, 36, 56, 38, 58, 42, 68, 66,
                                82, 86, 88, 92, 37, 57, 61, 63, 67, 87, 62]),
        ('hex', 216): np.array([0, 5, 35, 30, 180, 185, 215, 210, 1, 2, 3, 4,
                                6, 12, 18, 24, 36, 72, 108, 144, 11, 17, 23,
                                29, 41, 77, 113, 149, 34, 33, 32, 31, 71, 107,
                                143, 179, 66, 102, 138, 174, 181, 182, 183,
                                184, 186, 192, 198, 204, 191, 197, 203, 209,
                                214, 213, 212, 211, 7, 25, 28, 10, 13, 19, 26,
                                27, 22, 16, 9, 8, 14, 20, 21, 15, 37, 40, 148,
                                145, 38, 39, 76, 112, 147, 146, 109, 73, 74,
                                75, 111, 110, 42, 150, 168, 60, 78, 114, 156,
                                162, 132, 96, 54, 48, 84, 120, 126, 90, 47, 65,
                                173, 155, 53, 59, 101, 137, 167, 161, 119, 83,
                                89, 95, 131, 125, 70, 67, 175, 178, 69, 68,
                                103, 139, 176, 177, 142, 106, 105, 104, 140,
                                141, 187, 190, 208, 205, 188, 189, 196, 202,
                                207, 206, 199, 193, 194, 195, 201, 200, 43, 46,
                                64, 61, 151, 154, 172, 169, 44, 45, 49, 55, 79,
                                115, 52, 58, 82, 118, 63, 62, 100, 136, 97,
                                133, 152, 153, 157, 163, 160, 166, 171, 170,
                                50, 56, 57, 51, 80, 81, 117, 116, 85, 121, 127,
                                91, 88, 94, 130, 124, 99, 98, 134, 135, 158,
                                159, 165, 164, 86, 87, 93, 92, 122, 123, 129,
                                128])}

    pyfr_to_gmsh = {k: np.argsort(v, kind='heapsort') for k, v in
                    gmsh_to_pyfr.iteritems()}
