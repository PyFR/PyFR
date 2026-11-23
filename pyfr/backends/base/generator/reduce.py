from dataclasses import replace

from pyfr.dsl.nodes import Block, DSLCall, ExprStmt, Int, VarDecl
from pyfr.dsl.rewriter import parse_program


class ScratchSlots:
    def __init__(self):
        # Slots of each type needed by the most demanding exchange
        self._counts = {}

    def reserve(self, types):
        # Name a slot of each type, reusing the slots between exchanges
        names = []
        for i, t in enumerate(types):
            j = types[:i].count(t)
            names.append(f'_rs_{t}_{j}')
            self._counts[t] = max(self._counts.get(t, 0), j + 1)

        return names

    @property
    def names(self):
        return [d.name for d in self.decls(1)]

    def decls(self, n):
        # Declare each slot with an entry per lane of the group
        return [VarDecl(t, f'_rs_{t}_{j}', [Int(n)], shared=True)
                for t, cnt in self._counts.items() for j in range(cnt)]


def _combine_partners(names, types, read):
    # Read the partner value _p_v of each variable v and @combine() them
    code = ['{']
    code += [f'{t} _p_{v} = {read(i, v)};'
             for i, (v, t) in enumerate(zip(names, types))]
    code.append('@combine();\n}')

    return code


def _exchange(code, combine):
    # Splice a copy of the combine step into each exchange
    marker = ExprStmt(DSLCall('combine', []))

    def splice(node):
        match node:
            case Block(body) if marker in body:
                i = body.index(marker)
                body = body[:i] + combine + body[i + 1:]

                return replace(node, body=body)
            case _:
                return node.map(splice)

    return [splice(s) for s in parse_program('\n'.join(code)).body]


def butterfly(names, types, slots, size, combine, stride=1):
    # Exchange with each partner of the lane through its shared slot
    code = [f'for (int _m = {stride*size // 2}; _m >= {stride}; _m >>= 1)',
            '{']
    code += [f'{s}[@lidx] = {v};' for v, s in zip(names, slots)]
    code.append('@sync();')
    code += _combine_partners(names, types,
                              lambda i, v: f'{slots[i]}[@lidx ^ _m]')
    code.append('@sync();\n}')

    return _exchange(code, combine)


def gather(names, types, slots, size, combine, stride=1):
    # Gather the slots of the other lanes of the group into the lane
    n = stride*size
    code = [f'{s}[@lidx] = {v};' for v, s in zip(names, slots)]
    code.append('@sync();')
    code.append(f'for (int _m = {stride}; _m < {n}; _m += {stride})')
    code += _combine_partners(names, types,
                      lambda i, v: f'{slots[i]}[(@lidx + _m) % {n}]')
    code.append('@sync();')

    return _exchange(code, combine)
