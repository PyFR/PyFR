#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import uuid

from argparse import ArgumentParser, FileType, ArgumentTypeError
from collections import defaultdict, OrderedDict

import numpy as np

from pyfr.bases import get_std_ele_by_name
from pyfr.readers import get_reader_by_name, get_reader_by_extn, BaseReader
from pyfr.util import all_subclasses

# TODO: This needs to be sorted out.  "My eyes!  The goggles do nothing!"

def _gen_pdef(m_cdef, m_ddef, ndim):
    '''Returns two arrays, which specify the domain and
    discretised properties of each partition. These arrays are
    p_cdef[ndim,2,npartition] and p_ddef[ndim,npartition], which
    contain the min & max co-ordinates and the number of elements
    respectively, in each dimension and partition.'''

    p_dif = np.empty(m_cdef.shape[0])   #Vector of partition strides
    p_dif[:] = (m_cdef[:,1] - m_cdef[:,0]) / m_ddef[:,1]

    p_cdef = np.tile(np.vstack((m_cdef[:,0], (m_cdef[:,0] + p_dif[:]))).T,
                     tuple(m_ddef[::-1,-1]) + (1, 1))   #Tile standard part.

    offs = [(np.arange(m_ddef[i,1])*p_dif[i]) for i in xrange(ndim-1,-1,-1)]
               #List of partition offsets from std. across dims [z],y,x

    for dim, off in enumerate(np.ix_(*offs)):
        p_cdef[...,-dim-1,:] += off[...,None]   #Add part offsets to std.

    p_cdef = p_cdef.reshape(-1,ndim,2).transpose(1,2,0)
                #Concatenate partitions to 1D list, transpose to last axis

    npart = p_cdef.shape[2]
    p_ddef = np.zeros((ndim, npart), dtype=int)
    p_ddef[...] = m_ddef[:,0,None]

    return p_cdef, p_ddef, npart


def _gen_part_spts(p_cdef, p_ddef, npart, ndim, sord, ele_type):
    '''Returns the m_spts dictionary, which contains references to
    arrays of element shape points for each partition. The element
    shape point (spt) arrays are of dimension [nspts,nelements,ndim].
    '''

    # Get standard element
    std_ele = np.array(get_std_ele_by_name(ele_type, sord), dtype=float)

    std_ele += 1
    std_ele *= 0.5  #More convenient form
    ele_strd = np.zeros((ndim, npart))  #x,y,[z] size of elements by partition
    ele_strd[:,:] = (p_cdef[:,1,:] - p_cdef[:,0,:]) / p_ddef[:,:]

    m_spts = OrderedDict()
    for p in xrange(npart):
        p_name = 'spt_%s_p%d' % (ele_type, p)  #Spt array name for m_spts dict
        offs = [(np.arange(p_ddef[i,p])*ele_strd[i,p]) for
                i in xrange(ndim-1,-1,-1)]  #Offset of each ele from std.
        p_spts = np.tile((std_ele[...]*ele_strd[:,p]),
                         (tuple(p_ddef[::-1,p])+(1,1)))  #tile std. ele. spts

        for dim, off in enumerate(np.ix_(*offs)):
            p_spts[...,-dim-1] += (off[...,None] + p_cdef[-dim-1,0,p])
                #Add offset of each ele. to array of std. ele. spts.

        m_spts[p_name] = p_spts.reshape(-1,p_spts.shape[-2],ndim).swapaxes(0,1)
            #Reference arrays of ele spts by partition to a dictionary.

    return m_spts


def _gen_con(m_ddef, p_ddef, npart, ndim, ele_type):
    '''Returns the cons dictionary, which contains the element
    connectivity in each partition, and between partitions.
    Internal partition connectivity is given by a recarray of
    dimension [n_connections,2], and has naming of the form
    "con_p<n>", where <n> is the partition number.
    The recarray fields are of form:
    [<interface type>, <partition element #>, <element_face #>,
    <rotation tag>]. The second dimension of the array, of size
    two, defines the left and right of the interface.
    Inter-partition connectivity is similar, except the left and
    right of the interface is defined in separate files, analogous
    to the second dimension of the internal connectivity array:
    con_p<n>p<n1> and con_p<n1>p<n>, are both of dimension
    [n_connections]. <n> defines the partition in which the elements
    reside, and <n1> the partition to which they connect.
    '''

    p_num = np.arange(npart).reshape(tuple(m_ddef[::-1,1]))
                                    #Physical location of partitions
    cons = OrderedDict()

    rot_tags = np.array([1,0])  #Defines ele. rotation tags
    if ndim == 3:
        face_nums = np.array([[4, 2],[1, 3],[0, 5]])  #Hex face numbering
    else:
        face_nums = np.array([[3,1],[0,2]])  #Quad face numbering

    for p in xrange(npart):
        n = 'con_p%d' % (p) #Internal connectivity array name
        e_num = np.arange(np.product(p_ddef[:,p])).reshape(
                                     tuple(p_ddef[::-1,p]))  #Element location

        #Builds internal partition connectivity array, with periodic boundaries
        e_con = np.recarray((tuple(p_ddef[::-1,p])+(ndim,2)),
                           dtype='|S5,i4,i1,i1')
        e_con[...]['f0'] = ele_type
        e_con[...]['f2'] = face_nums[...]
        e_con[...]['f3'] = rot_tags[:]
        e_con[...,0]['f1'] = e_num[...,None]
        for dim in xrange(ndim):  #Shifts element locations to find neighbour
            e_con[...,dim,1]['f1'] = np.roll(e_num, 1, axis=ndim-1-dim)

        del_e_con = []
        ind = (Ellipsis,)*(ndim-1) + (0,)  #Useful index
        for dim in xrange(ndim-1,-1,-1):  #Generates inter-partition
            idm = ndim -1 -dim  #Compensates for x fastest counting order
            if int(m_ddef[idm,1]) > 1:
                p_loc = np.array(np.where(p_num==p)).reshape(-1)  #locate part
                p_nex = np.roll(p_num, 1, axis=dim)[tuple(p_loc)]
                                                #locate neighbours of partition
                p_bef = np.roll(p_num, -1, axis=dim)[tuple(p_loc)]

                n0 = 'con_p%dp%d' % (p, p_nex)  #Inter-partition connectivity
                n1 = 'con_p%dp%d' % (p, p_bef)

                if p_nex == p_bef:   #Periodic BC for two partitions in a dim
                    f = np.array([0,1])  #arrange part 1 for periodic BC
                    if 'con_p%dp%d' % (p_nex, p) in cons: f = np.flipud(f)
                    cons[n0] = np.hstack((e_con[(ind+(idm,f[0]))].reshape(-1),
                                          e_con[(ind+(idm,f[1]))].reshape(-1)))

                else:
                    #Uses internal periodic partition connectivity
                    #to generate periodic inter-partition connect.
                    cons[n0] = e_con[(ind+(idm,0))].reshape(-1)
                    cons[n1] = e_con[(ind+(idm,1))].reshape(-1)

                del_e_con.append(e_num[ind].flatten() * ndim + (idm))
                    #Accumulates internal periodic connectivity to delete

            ind  = ind[1:] + (ind[0],)  #Increments index

        if len(del_e_con) > 0:
            cons[n] = np.delete(e_con.reshape(-1,2).T, np.hstack(del_e_con), 1)
                #Deletes internal periodic connectivity
        else:
            cons[n] = e_con.reshape(-1,2).T

    return cons

def _curve_mesh(mesh, m_cdef, npart, ele_type, curve):
    '''Curves the hyper-rectangular mesh generated by
    _gen_part_spts(). The function applies a sinusoidal
    distortion along the axis specified by curve[0] in
    the curve[1] axis direction. curve[2] defines the
    magnitude of the bump as percentage of domain width
    in the curve[1] axis direction. curve[3] defines
    the number of periods of the bump along axis curve[0].'''

    ax, cx = curve[0], curve[1]
    if ax >= m_cdef.shape[0] or cx >= m_cdef.shape[0] or cx == ax:
        raise ArgumentError('invalid combination of axes for domain curving')
    if curve[2] < 0 or curve[3] < 0:
        raise ArgumentError('invalid mesh curving parameters')

    amp = (m_cdef[cx,1] - m_cdef[cx,0]) * 0.01 * curve[2] #curve amplitude
    rad = curve[3] * 2.0 * np.pi / float(m_cdef[ax,1] - m_cdef[ax,0])  #ax->rad

    for p in xrange(npart):
        n = 'spt_%s_p%d' % (ele_type, p)
        spts = mesh[n]
        spts[...,cx] += amp * np.sin(spts[...,ax] * rad)  #Curve mesh
        mesh[n] = spts

def process_gen(args):
    '''Generates a high-order multi-partition test mesh compatible
    with PyFR.

    '''

    ndim = len(args.dmin)
    if ndim < 2 or ndim > 3:
        raise ArgumentError('domain must have dimensions "x y [z]"')
    if len(args.dmax) != ndim or len(args.epp) != ndim or len(args.prt) !=ndim:
        raise ArgumentError('dimension mismatch in mesh definition')

    m_cdef = np.vstack((args.dmin, args.dmax)).T
    m_ddef = np.vstack((args.epp, args.prt)).T
    sord, ele_type = args.sord, args.ele_type

    if (m_cdef[:,1]-m_cdef[:,0]).any() <= 0 or m_ddef.any() <= 0:
        raise ArgumentError('invalid mesh domain or discretisation')

    p_cdef, p_ddef, npart = _gen_pdef(m_cdef, m_ddef, ndim)

    mesh = _gen_part_spts(p_cdef, p_ddef, npart, ndim, sord, ele_type)

    cons = _gen_con(m_ddef, p_ddef, npart, ndim, ele_type)

    if args.curve != None:
        _curve_mesh(mesh, m_cdef, npart, ele_type, args.curve)

    mesh_uuid = str(uuid.uuid4())  #Mesh unique identifier.
    mesh['mesh_uuid'] = mesh_uuid
    mesh.update(cons)
    print 'Mesh generation complete; writing to disk.'
    np.savez_compressed(args.meshfile, **mesh)  #save to disk (compressed)

def process_convert(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh)
    else:
        extn = os.path.splitext(args.inmesh.name)[1]
        reader = get_reader_by_extn(extn, args.inmesh)

    # Get the mesh in the PyFR format
    mesh = reader.to_pyfrm()

    # Save to disk
    np.savez(args.outmesh, **mesh)

def main():
    ap = ArgumentParser(prog='pyfr-mesh', description='Generates and '
                        'manipulates PyFR mesh files')

    sp = ap.add_subparsers(help='sub-command help')

    # Mesh format conversion
    ap_convert = sp.add_parser('convert', help='convert --help')
    ap_convert.add_argument('inmesh', type=FileType('r'),
                            help='Input mesh file')
    ap_convert.add_argument('outmesh', type=FileType('wb'),
                            help='Output PyFR mesh file')
    types = [cls.name for cls in all_subclasses(BaseReader)]
    ap_convert.add_argument('-t', dest='type', choices=types, required=False,
                            help='Input file type; this is usually inferred '
                            'from the extension of inmesh')
    ap_convert.set_defaults(process=process_convert)

    ap_gen = sp.add_parser('gen', help='gen --help')
    ap_gen.add_argument('ele_type', choices=['quad', 'tri', 'hex', 'tet',
                        'prism', 'pyrm'], help='select element type')
    ap_gen.add_argument('meshfile', type=FileType('w'), help='mesh filename')
    ap_gen.add_argument('-sord', required=True, type=int,
                    help='element shape order')
    ap_gen.add_argument('-dmin', required=True, type=float, nargs='+',
                    help='minimum co-ordinates of the mesh domain: x y [z]')
    ap_gen.add_argument('-dmax', required=True, type=float, nargs='+',
                    help='maximum co-ordinates of the mesh domain: x y [z]')
    ap_gen.add_argument('-epp', required=True, type=int, nargs='+',
                    help='number of elements per partition in dims: x y [z]')
    ap_gen.add_argument('-prt', required=True, type=int, nargs='+',
                    help='number of partitions in dims: x y [z]')
    ap_gen.add_argument('-curve', type=int, required=False, nargs=4,
                    help='apply a sinusoidal offset to the domain of period \
                    CURVE3, in the CURVE0 axis direction, along the CURVE1 \
                    axis. Offset amplitude is defined as percentage of offset \
                    domain width by CURVE2. E.g. "-curve 1 2 30 4" curves the \
                    domain in the y-direction along the z-axis, with 4 \
                    periods along the z axis. Offset amplitude is 0.3 * domain\
                    width in the y-direction.')
    ap_gen.set_defaults(process=process_gen)

    args = ap.parse_args()
    args.process(args)

if __name__ == '__main__':
    main()
