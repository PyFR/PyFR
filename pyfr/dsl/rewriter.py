from pyfr.dsl.lexer import Lexer
from pyfr.dsl.nodes import (DSLVar, Float, Transformer, Var, VarDecl,
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
            case Var(name) | DSLVar(name) if name in renames:
                return renames[name]
            case VarDecl(vt, n, sz, init, const=c) if n in renames:
                r = renames[n]

                # Only a plain variable can stand in for a declaration
                if isinstance(r, Var):
                    if sz:
                        sz = [rename(s) for s in sz]

                    init = rename(init) if init else None
                    return VarDecl(vt, r.name, sz, init, const=c)
                else:
                    return node.map(rename)
            case _:
                return node.map(rename)

    return rename(ast)


class Rewriter(Transformer):
    def __init__(self, rules):
        self.rules = rules

    def transform(self, node, args):
        self.args = args
        node = self.visit(node)

        # Any argument reference which survives had no applicable rule
        for n in walk_ast(node):
            if isinstance(n, Var) and n.name in args:
                raise ValueError(f'No matching rule for {n.name}')

        return node

    def visit(self, node):
        # Check the arity is consistent
        n, ix = unwrap_index(node)
        args = self.args
        if n in args and len(ix) > args[n].nidx:
            raise ValueError(f'Too many indices for {n}')

        return super().visit(node)

    def rewrite(self, node):
        # Extract the name and indices, then look up a matching arg
        n, ix = unwrap_index(node)
        if n in self.args:
            arg = self.args[n]
            for r in self.rules:
                if (result := r(n, ix, arg)) is not None:
                    return result

        return node


class FloatSuffixer(Transformer):
    def rewrite(self, node):
        if isinstance(node, Float) and not node.suffix:
            return Float(node.value, 'f', node.ishex)
        else:
            return node
