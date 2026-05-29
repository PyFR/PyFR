from pyfr.util import paren_depths


def split_components(expr):
    # Split on top-level commas; respect bracket/paren/brace nesting
    parts, buf = [], []
    for c, d in paren_depths(expr):
        if c == ',' and d == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(c)

    parts.append(''.join(buf).strip())

    return parts


def bp_key(k):
    return k.replace('_', '/').replace('-', '_')
