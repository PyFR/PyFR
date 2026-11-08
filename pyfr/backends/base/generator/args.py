import re


class Arg:
    def __init__(self, name, spec):
        self.name = name

        specptn = r'''
            (?:(in|inout|out)\s+)?                            # Intent
            (?:((?:broadcast-col\s+)?view(?:\(\d+\))?|        # Attrs
                broadcast(?:-row|-col)?|mpi|scalar)\s+)?
            (?:reduce\((min|max|sum)\)\s+)?                   # Reduction
            ([A-Za-z_]\w*)                                    # Data type
            ((?:\[\d+\]){0,2})                                # Dimensions
        '''
        dimsptn = r'(?<=\[)\d+(?=\])'

        # Parse our specification
        if not (m := re.fullmatch(specptn, spec, re.X)):
            raise ValueError('Invalid argument specification')

        g = m.groups()

        # Properties
        self.intent = g[0] or 'in'
        self.attrs = g[1] or ''
        self.reduceop = g[2]
        self.dtype = g[3]
        self.cdimstr = g[4]

        # Dimension
        self.cdims = [int(d) for d in re.findall(dimsptn, g[4])]
        self.ncdim = len(self.cdims)

        # View stride (for multi-indexed views: view(N))
        if m := re.search(r'view\((\d+)\)', self.attrs):
            self.viewstride = int(m[1])
        else:
            self.viewstride = 1

        # Multi-indexed views take an extra leading index in the body
        self.ismulti = self.viewstride > 1
        self.nidx = self.ncdim + self.ismulti

        # Attributes
        self.isbroadcast = self.attrs == 'broadcast'
        self.isbroadcastr = self.attrs == 'broadcast-row'
        self.isbroadcastc = 'broadcast-col' in self.attrs
        self.ismpi = self.attrs == 'mpi'
        self.isview = 'view' in self.attrs
        self.isscalar = self.attrs == 'scalar'
        self.isvector = not self.isscalar
        self.isreduce = bool(self.reduceop)

        # Validation
        if (self.attrs.startswith('broadcast') and
            self.intent != 'in' and not self.isreduce):
            raise ValueError('Broadcast arguments must be of intent in')
        if self.isbroadcast and self.ncdim != 2:
            raise ValueError('Broadcasts must have two dimensions')
        if self.isbroadcastr and self.ncdim != 1:
            raise ValueError('Row broadcasts must have one dimension')
        if self.isreduce and self.intent != 'out':
            raise ValueError('Reduction arguments must be of intent out')
        if self.isreduce and self.dtype != 'fpdtype_t':
            raise ValueError('Reduction arguments must be of type fpdtype_t')
        if self.isreduce and self.isscalar:
            raise ValueError('Scalar arguments can not be reduced')
        if self.isreduce and self.ncdim and not self.isbroadcastc:
            raise ValueError('Non-broadcast-col reduction args must be scalar')
        if self.isscalar and self.dtype not in ('fpdtype_t', 'ixdtype_t'):
            raise ValueError('Scalar arguments must be fpdtype_t or ixdtype_t')
