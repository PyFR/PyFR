from dataclasses import replace
import re

from pyfr.dsl.nodes import (BINARY_OPS, COMPOUND_ASSIGN, POSTFIX_OPS,
                            PRECEDENCE, RIGHT_ASSOC, UNARY_OPS, ArrayInit,
                            Assign, Binary, Block, Break, Call, Cast, Comment,
                            Continue, DoWhile, DSLCall, DSLVar, ExprStmt,
                            Float, For, If, Index, Int, MultiDecl, Postfix,
                            Program, Region, String, Ternary, Typedef, Unary,
                            Var, VarDecl, While)


class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.tokens = iter(lexer)
        self.pcomments = []
        self.tok = None
        self.advance()

    def advance(self):
        tok = self.tok

        # Bank any comment trivia for the next statement position
        while (nxt := next(self.tokens, tok)).kind == 'COMMENT':
            self.pcomments.append(nxt.value)

        self.tok = nxt
        return tok

    def take_comments(self):
        comments, self.pcomments = self.pcomments, []

        return tuple(comments)

    def drain_comments(self):
        return [Comment(k, t) for k, t in self.take_comments()]

    def error(self, msg, tok):
        return self.lexer.error(f'{msg} {tok.value!r}', tok.pos)

    def accept(self, kind):
        if self.tok.kind == kind:
            return self.advance()

    def expect(self, kind):
        if self.tok.kind != kind:
            raise self.error(f'Expected {kind}, got', self.tok)

        return self.advance()

    def _parse_decl_head(self):
        is_const = bool(self.accept('const'))

        return is_const, self.expect('TYPE')

    def parse(self):
        body = []
        while self.tok.kind != 'EOF':
            body.append(self.statement())

        body += self.drain_comments()
        return Program(body)

    def typedef_stmt(self):
        self.advance()
        old = self.expect('TYPE').value
        new = self.expect('IDENT').value

        # Have the lexer yield the new name as a type from here on
        self.lexer.kinds[new] = 'TYPE'
        self.expect(';')

        return Typedef(old, new)

    def statement(self):
        comments = self.take_comments()

        return replace(self._statement(), comments=comments)

    def _statement(self):
        match self.tok.kind:
            case 'TYPE' | 'const':
                return self.declaration()
            case 'typedef':
                return self.typedef_stmt()
            case 'if':
                return self.if_stmt()
            case 'while':
                return self.while_stmt()
            case 'for':
                return self.for_stmt()
            case 'do':
                return self.do_while_stmt()
            case 'break' | 'continue':
                tok = self.advance()
                self.expect(';')
                return Break() if tok.kind == 'break' else Continue()
            case 'PRAGMA':
                return self.pragma_stmt()
            case '{':
                return self.block()
            case ';':
                self.advance()
                return ExprStmt(None)
            case _:
                expr = self.expression(0)
                self.expect(';')
                return ExprStmt(expr)

    def block(self):
        self.expect('{')

        stmts = []
        while self.tok.kind != '}':
            stmts.append(self.statement())

        stmts += self.drain_comments()
        self.expect('}')

        return Block(stmts)

    def pragma_stmt(self):
        tok = self.advance()

        # Our pragmas scope a region kind to the block which follows
        name, _, rest = tok.value.partition(' ')
        m = re.match(r'([\w-]+)\s*(?:\((.*)\))?$', rest.strip())
        if name != 'pyfr' or m is None:
            raise self.error('Unsupported pragma', tok)

        # Split any parenthesised region arguments on commas
        args = tuple(a.strip() for a in m[2].split(',')) if m[2] else ()

        return Region(m[1], self.block().body, args)

    def declaration(self):
        decl = self._decl_list()
        self.expect(';')

        return decl

    def _decl_list(self):
        is_const, type_tok = self._parse_decl_head()

        decls = [self._declarator(is_const, type_tok.value)]
        while self.accept(','):
            decls.append(self._declarator(is_const, type_tok.value))

        return decls[0] if len(decls) == 1 else MultiDecl(decls)

    def _declarator(self, is_const, vtype):
        name = self.expect('IDENT').value

        sizes = []
        while self.accept('['):
            if self.tok.kind == ']':
                sizes.append(None)
            else:
                sizes.append(self.expression(0))
            self.expect(']')

        if self.accept('='):
            # Check if it is an array-style init
            if self.tok.kind == '{':
                init = self.array_initialiser()
                if sizes and sizes[0] is None:
                    sizes[0] = Int(len(init.values))
                self._check_initialiser(name, sizes, init)
            else:
                init = self.expression(PRECEDENCE[','])
        else:
            init = None

        # Ensure all array extents are sized
        if None in sizes:
            raise SyntaxError(f'Unsized extent on array {name!r}')

        return VarDecl(vtype, name, sizes or None, init, const=is_const)

    def _check_initialiser(self, name, sizes, init):
        # Skip checks for scalars and non-trivial extents such as x[1+1]
        if not sizes or not all(isinstance(s, Int) for s in sizes):
            return

        def check(vals, dims):
            if len(vals) > dims[0].value:
                raise SyntaxError(f'Too many initialisers for array {name!r}')

            for v in vals:
                nested = isinstance(v, ArrayInit)
                if nested and len(dims) > 1:
                    check(v.values, dims[1:])
                # Permit the {0} idiom for zero-filling a nested array
                elif nested or (len(dims) > 1 and v != Int(0)):
                    raise SyntaxError(f'Malformed initialiser on {name!r}')

        check(init.values, sizes)

    def array_initialiser(self):
        self.expect('{')

        values = []
        while self.tok.kind != '}':
            if self.tok.kind == '{':
                values.append(self.array_initialiser())
            else:
                values.append(self.expression(PRECEDENCE[','] + 1))

            if self.tok.kind != '}':
                self.expect(',')

        self.expect('}')

        return ArrayInit(values)

    def _parse_cond_stmt(self):
        self.expect('(')
        cond = self.expression(0)
        self.expect(')')

        return cond

    def if_stmt(self):
        self.advance()
        cond = self._parse_cond_stmt()
        then = self.statement()
        else_ = self.statement() if self.accept('else') else None

        return If(cond, then, else_)

    def while_stmt(self):
        self.advance()

        return While(self._parse_cond_stmt(), self.statement())

    def for_stmt(self):
        self.advance()
        self.expect('(')

        if self.tok.kind == ';':
            init = None
        elif self.tok.kind in ('TYPE', 'const'):
            init = self._decl_list()
        else:
            init = self.expression(0)

        self.expect(';')
        cond = None if self.tok.kind == ';' else self.expression(0)
        self.expect(';')
        step = None if self.tok.kind == ')' else self.expression(0)
        self.expect(')')

        return For(init, cond, step, self.statement())

    def do_while_stmt(self):
        self.advance()
        body = self.statement()
        self.expect('while')
        cond = self._parse_cond_stmt()
        self.expect(';')

        return DoWhile(body, cond)

    def expression(self, rbp=0):
        left = self.nud(self.advance())

        # Only symbol tokens may act as operators
        while PRECEDENCE.get(self.tok.kind, 0) > rbp:
            left = self.led(self.advance(), left)

        return left

    def nud(self, tok):
        match tok.kind:
            case 'INT' | 'FLOAT':
                cls = Int if tok.kind == 'INT' else Float
                return cls(*tok.value)
            case 'STRING':
                return String(tok.value)
            case 'IDENT':
                if self.accept('('):
                    args = self._parse_call_args()

                    # Canonicalise pow
                    if tok.value == 'pow':
                        return Binary('**', *args)
                    else:
                        return Call(Var(tok.value), args)
                else:
                    return Var(tok.value)
            case 'DSL':
                if self.accept('('):
                    return DSLCall(tok.value, self._parse_call_args())
                else:
                    return DSLVar(tok.value)
            case op if op in UNARY_OPS:
                return Unary(op, self.expression(PRECEDENCE['unary']))
            case '(':
                # A leading type makes this a cast rather than a group
                if type_tok := self.accept('TYPE'):
                    self.expect(')')
                    return Cast(type_tok.value,
                                self.expression(PRECEDENCE['unary']))
                else:
                    expr = self.expression(0)
                    self.expect(')')
                    return expr
            case _:
                raise self.error('Invalid expression start', tok)

    def _parse_call_args(self):
        args = []
        if self.tok.kind != ')':
            while True:
                args.append(self.expression(PRECEDENCE[','] + 1))
                if self.tok.kind == ')':
                    break

                self.expect(',')

        self.expect(')')

        return args

    def led(self, tok, left):
        match tok.kind:
            case op if op in POSTFIX_OPS:
                return Postfix(op, left)
            case '[':
                index = self.expression(0)
                self.expect(']')
                return Index(left, index)
            case '(':
                return Call(left, self._parse_call_args())
            case '?':
                true_expr = self.expression(0)
                self.expect(':')

                # The else branch of a ternary is a conditional-expression
                false_expr = self.expression(PRECEDENCE['?'] - 1)
                return Ternary(left, true_expr, false_expr)
            case ',':
                return Binary(',', left, self.expression(PRECEDENCE[',']))
            case op if op in COMPOUND_ASSIGN:
                right = self.expression(PRECEDENCE[op] - 1)
                return Assign(left, Binary(op[:-1], left, right))
            case '=':
                return Assign(left, self.expression(PRECEDENCE['='] - 1))
            case op if op in BINARY_OPS:
                prec = PRECEDENCE[op]
                rbp = prec - 1 if op in RIGHT_ASSOC else prec
                return Binary(op, left, self.expression(rbp))
            case _:
                raise self.error('Invalid operator', tok)
