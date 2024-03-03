def call_(obj, name_, **kwargs):
    keys = list(kwargs)
    keys[0] = keys[0][0].upper() + keys[0][1:]
    meth = name_ + '_'.join(keys) + '_'

    return getattr(obj, meth)(*kwargs.values())


def init_(cls, **kwargs):
    return call_(cls.alloc(), 'initWith', **kwargs)
