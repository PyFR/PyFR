# -*- coding: utf-8 -*-

def dot(context, l, r, len='ndims'):
    # Format ({})*({}) => (lvar[{0}])*(rvar[{0}])
    lr = '({})*({})'.format(l, r)

    # Run over each dimension
    nd = context.get(len)
    return '(' + ' + '.join(lr.format(k) for k in range(nd)) + ')'


def vlen(context, v, len='ndims'):
    return dot(context, v, v, len)
