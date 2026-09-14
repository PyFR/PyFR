import re

from pyfr.dsl.nodes import (ArrayInit, Assign, Binary, Block, Break, Call,
                            Cast, Comment, Continue, DoWhile, DslCall, DslVar,
                            ExprStmt, Float, For, If, Index, Int, MultiDecl,
                            Postfix, Program, Region, String, Ternary, Typedef,
                            Unary, Var, VarDecl, While, BINARY_OPS,
                            COMPOUND_ASSIGN, POSTFIX_OPS, PRECEDENCE,
                            RIGHT_ASSOC, UNARY_OPS)


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0
        self.pcomments = []
        self.scopes = [{}]
        self._typedefs = set()

    def peek(self):
        return self.tokens[self.i]

    def advance(self):
        tok = self.peek()
        self.i += 1

        # Bank any comment trivia for the next statement position
        if tok.comments:
            self.pcomments += tok.comments

        return tok

    def take_comments(self):
        tok = self.peek()
        comments, self.pcomments = self.pcomments + list(tok.comments), []
        tok.comments = ()

        return tuple(comments)

    def drain_comments(self):
        return [Comment(k, t) for k, t in self.take_comments()]

    def accept(self, value):
        if self.peek().value == value:
            return self.advance()

    def expect(self, kind, value=None):
        tok = self.peek()
        if tok.kind != kind or (value is not None and tok.value != value):
            raise SyntaxError(f'Expected {kind} {value}, got {tok}')

        return self.advance()

    def _is_type(self, tok):
        return tok.kind == 'TYPE' or tok.value in self._typedefs

    def _is_decl_start(self, tok):
        return self._is_type(tok) or tok.value == 'const'

    def _parse_decl_head(self):
        is_const = bool(self.accept('const'))

        tok = self.peek()
        if not self._is_type(tok):
            raise SyntaxError(f'Expected type, got {tok}')

        return is_const, self.advance()

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()

    def parse(self):
        body = []
        while self.peek().kind != 'EOF':
            body.append(self.statement())

        body += self.drain_comments()
        return Program(body)

    def typedef_stmt(self):
        self.advance()
        tok = self.peek()
        if not self._is_type(tok):
            raise SyntaxError(f'Expected type after typedef, got {tok}')

        old = self.advance().value
        new = self.expect('IDENT').value
        self.expect('SYMBOL', ';')
        self._typedefs.add(new)

        return Typedef(old, new)

    def statement(self):
        comments = self.take_comments()
        stmt = self._statement()
        stmt.comments = comments

        return stmt

    def _statement(self):
        tok = self.peek()
        match tok.kind, tok.value:
            case ('TYPE', _) | ('KEYWORD', 'const'):
                return self.declaration()
            case ('KEYWORD', 'typedef'):
                return self.typedef_stmt()
            case ('KEYWORD', 'if'):
                return self.if_stmt()
            case ('KEYWORD', 'while'):
                return self.while_stmt()
            case ('KEYWORD', 'for'):
                return self.for_stmt()
            case ('KEYWORD', 'do'):
                return self.do_while_stmt()
            case ('KEYWORD', 'break' | 'continue'):
                self.advance()
                self.expect('SYMBOL', ';')
                return Break() if tok.value == 'break' else Continue()
            case ('KEYWORD', kw):
                p = f'{tok.line}:{tok.col}'
                raise SyntaxError(f'Unexpected keyword {kw!r} at {p}')
            case ('PRAGMA', _):
                return self.pragma_stmt()
            case ('SYMBOL', '{'):
                return self.block()
            case ('SYMBOL', ';'):
                self.advance()
                return ExprStmt(None)
            case ('IDENT', name) if name in self._typedefs:
                return self.declaration()
            case _:
                expr = self.expression(0)
                self.expect('SYMBOL', ';')
                return ExprStmt(expr)

    def block(self):
        self.expect('SYMBOL', '{')
        self.push_scope()

        stmts = []
        while self.peek().value != '}':
            stmts.append(self.statement())

        stmts += self.drain_comments()
        self.expect('SYMBOL', '}')
        self.pop_scope()

        return Block(stmts)

    def pragma_stmt(self):
        tok = self.advance()

        # Our pragmas scope a region kind to the block which follows
        name, _, rest = tok.value.partition(' ')
        m = re.match(r'([\w-]+)\s*(?:\((.*)\))?$', rest.strip())
        if name != 'pyfr' or m is None:
            p = f'{tok.line}:{tok.col}'
            raise SyntaxError(f'Unsupported pragma {tok.value!r} at {p}')

        # Split any parenthesised region arguments on commas
        args = tuple(a.strip() for a in m[2].split(',')) if m[2] else ()

        return Region(m[1], self.block().body, args)

    def declaration(self):
        decl = self._decl_list()
        self.expect('SYMBOL', ';')

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
            if self.peek().value == ']':
                sizes.append(None)
            else:
                sizes.append(self.expression(0))
            self.expect('SYMBOL', ']')

        if self.accept('='):
            # An unsized array takes its extent from the initialiser
            if self.peek().value == '{':
                init = self.array_initialiser()
                if sizes and sizes[0] is None:
                    sizes[0] = Int(len(init.values))
            else:
                init = self.expression(PRECEDENCE[','])
        else:
            init = None

        self.scopes[-1][name] = vtype

        return VarDecl(is_const, vtype, name, sizes or None, init)

    def array_initialiser(self):
        self.expect('SYMBOL', '{')

        values = []
        while self.peek().value != '}':
            if self.peek().value == '{':
                values.append(self.array_initialiser())
            else:
                values.append(self.expression(PRECEDENCE[','] + 1))

            if self.peek().value != '}':
                self.expect('SYMBOL', ',')

        self.expect('SYMBOL', '}')

        return ArrayInit(values)

    def _parse_cond_stmt(self):
        self.expect('SYMBOL', '(')
        cond = self.expression(0)
        self.expect('SYMBOL', ')')

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
        self.expect('SYMBOL', '(')

        tok = self.peek()
        if tok.value == ';':
            init = None
        elif self._is_decl_start(tok):
            init = self._decl_list()
        else:
            init = self.expression(0)

        self.expect('SYMBOL', ';')
        cond = None if self.peek().value == ';' else self.expression(0)
        self.expect('SYMBOL', ';')
        step = None if self.peek().value == ')' else self.expression(0)
        self.expect('SYMBOL', ')')

        return For(init, cond, step, self.statement())

    def do_while_stmt(self):
        self.advance()
        body = self.statement()
        self.expect('KEYWORD', 'while')
        cond = self._parse_cond_stmt()
        self.expect('SYMBOL', ';')

        return DoWhile(body, cond)

    def _op_prec(self, tok):
        # Only symbol tokens may act as operators
        if tok.kind == 'SYMBOL':
            return PRECEDENCE.get(tok.value, 0)
        else:
            return 0

    def expression(self, rbp=0):
        left = self.nud(self.advance())

        while self._op_prec(self.peek()) > rbp:
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
                    return DslCall(tok.value, self._parse_call_args())
                else:
                    return DslVar(tok.value)
            case 'SYMBOL' if tok.value in UNARY_OPS:
                return Unary(tok.value, self.expression(PRECEDENCE['unary']))
            case 'SYMBOL' if tok.value == '(':
                next_tok = self.peek()

                # A leading type makes this a cast rather than a group
                if self._is_type(next_tok):
                    type_tok = self.advance()
                    self.expect('SYMBOL', ')')
                    return Cast(type_tok.value,
                                self.expression(PRECEDENCE['unary']))
                else:
                    expr = self.expression(0)
                    self.expect('SYMBOL', ')')
                    return expr
            case _:
                raise SyntaxError(f'Invalid expression start {tok}')

    def _parse_call_args(self):
        args = []
        if self.peek().value != ')':
            while True:
                args.append(self.expression(PRECEDENCE[','] + 1))
                if self.peek().value == ')':
                    break

                self.expect('SYMBOL', ',')

        self.expect('SYMBOL', ')')

        return args

    def led(self, tok, left):
        match tok.value:
            case op if op in POSTFIX_OPS:
                return Postfix(op, left)
            case '[':
                index = self.expression(0)
                self.expect('SYMBOL', ']')
                return Index(left, index)
            case '(':
                return Call(left, self._parse_call_args())
            case '?':
                true_expr = self.expression(0)
                self.expect('SYMBOL', ':')

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
                raise SyntaxError(f'Invalid operator {tok.value}')
