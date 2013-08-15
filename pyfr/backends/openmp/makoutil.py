# -*- coding: utf-8 -*-

import pyfr.util as util


def dot(context, l, r, len='ndims'):
    # Format ({})*({}) => (lvar[{0}])*(rvar[{0}])
    lr = '({})*({})'.format(l, r)

    # Run over each dimension
    nd = context.get(len)
    return '(' + ' + '.join(lr.format(k) for k in range(nd)) + ')'


def vlen(context, v, len='ndims'):
    return dot(context, v, v, len)


def arr_args(context, name, shape, dtype='dtype', const=False):
    if any(s > 10 for s in shape):
        raise ValueError('Unsupported dimensions')

    # Base argument part (e.g, const double *restrict u)
    base = '{} {} *restrict {}'.format('const' if const else '',
                                       context.get(dtype), name)

    # Elements of the array
    base += '{}'*len(shape)
    return ', '.join(base.format(*ix) for ix in util.ndrange(*shape))


def arr_align(context, name, shape):
    base = 'ASSUME_ALIGNED({}{});'.format(name, '{}'*len(shape))

    # Format, join and trim the trailing ';'
    return ' '.join(base.format(*ix) for ix in util.ndrange(*shape))[:-1]


def ndrange(context, *args):
    return util.ndrange(*args)
