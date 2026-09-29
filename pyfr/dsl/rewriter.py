from pyfr.dsl.lexer import Lexer
from pyfr.dsl.nodes import (DslVar, Float, Var, VarDecl, map_ast,
                            unwrap_index, walk_ast)
from pyfr.dsl.parser import Parser


def parse_expr(code, lexer=Lexer):
    p = Parser(lexer(code).tokenise())
    expr = p.expression(0)

    # Trailing tokens would otherwise be silently discarded
    if (tok := p.peek()).kind != 'EOF':
        raise SyntaxError(f'Unexpected {tok.value!r} in expression: {code}')

    return expr


def parse_stmt(code):
    return Parser(Lexer(code).tokenise()).statement()


def parse_program(code, types=()):
    return Parser(Lexer(code, types=types).tokenise()).parse()


def rename_vars(ast, renames):
    def rename(node):
        match node:
            case Var(name) | DslVar(name) if name in renames:
                return renames[name]
            case VarDecl(c, vt, n, sz, init) if n in renames:
                r = renames[n]

                # Only a plain variable can stand in for a declaration
                if isinstance(r, Var):
                    if sz:
                        sz = [rename(s) for s in sz]

                    init = rename(init) if init else None
                    return VarDecl(c, vt, r.name, sz, init)
                else:
                    return map_ast(node, rename)
            case _:
                return map_ast(node, rename)

    return rename(ast)


class Rewriter:
    def __init__(self, rules):
        self.rules = rules

    def transform(self, node, args):
        node = self._transform(node, args)

        # Any argument reference which survives had no applicable rule
        for n in walk_ast(node):
            if isinstance(n, Var) and n.name in args:
                raise ValueError(f'No matching rule for {n.name}')

        return node

    def _transform(self, node, args):
        # Check the arity is consistent
        n, ix = unwrap_index(node)
        if n in args and len(ix) > args[n].nidx:
            raise ValueError(f'Too many indices for {n}')

        mapped = map_ast(node, lambda n: self._transform(n, args))

        # Extract the name and indices, then look up a matching arg
        n, ix = unwrap_index(mapped)
        if n is not None and n in args:
            arg = args[n]
            for r in self.rules:
                if (result := r(n, ix, arg)) is not None:
                    return result

        return mapped


class FloatSuffixer:
    def transform(self, node):
        if isinstance(node, Float) and not node.suffix:
            return Float(node.value, 'f', node.ishex)
        else:
            return map_ast(node, self.transform)
