from pyfr.dsl.nodes import (ArrayInit, Assign, Binary, Block, Break, Call,
                            Cast, Comment, Continue, DoWhile, DslCall, DslVar,
                            ExprStmt, Float, For, If, Index, Int, MultiDecl,
                            Postfix, Program, Region, String, Ternary, Typedef,
                            Unary, Var, VarDecl, While, COMPOUND_ASSIGN,
                            PRECEDENCE, RIGHT_ASSOC)


class CodeGenerator:
    _exprs = (Assign, Binary, Call, Cast, DslCall, DslVar, Float, Index, Int,
              Postfix, String, Ternary, Unary, Var)

    # Anything which is not an operator never needs parenthesising
    _non_op_prec = max(PRECEDENCE.values(), default=0) + 1

    # C spellings for the functions we also accept under their bare names
    _func_canon = {'abs': 'fabs', 'min': 'fmin', 'max': 'fmax'}

    # Binary operators which have a compound assignment form
    _compound = {op[:-1] for op in COMPOUND_ASSIGN}

    # Escape sequences for string literals
    _str_esc = str.maketrans({'\\': '\\\\', '"': '\\"', '\n': '\\n',
                              '\t': '\\t', '\r': '\\r'})

    # Region kinds to the text which opens them
    region_markers = {}

    def __init__(self):
        self.nind = 0
        self.ind = ' '*4

    def generate(self, ast):
        if isinstance(ast, Program):
            return self._generate_program(ast)
        elif isinstance(ast, self._exprs):
            return self._generate_expr(ast)
        else:
            return self._generate_statement(ast)

    # Lower a region kind to the text which opens it
    def gen_region_open(self, kind):
        return self.region_markers.get(kind)

    def _generate_region(self, node):
        marker = self.gen_region_open(node.kind)

        # Without a lowering, re-emit the pragma so the output re-parses
        if marker is None:
            if node.args:
                args = '(' + ', '.join(node.args) + ')'
            else:
                args = ''

            indent = self.nind*self.ind
            block = self._generate_block(node)
            return f'#pragma pyfr {node.kind}{args}\n{indent}{block}'
        else:
            return self._generate_block(node, marker)

    def _generate_program(self, ast):
        stmts = (self._gen_comments(s, '') + self._generate_statement(s)
                 for s in ast.body)
        return '\n'.join(stmts)

    # Render the leading comment trivia of a node as prefix lines
    def _gen_comments(self, node, indent):
        cs = (self._generate_statement(Comment(*c)) for c in node.comments)
        return ''.join(f'{c}\n{indent}' for c in cs)

    def _generate_statement(self, stmt):
        match stmt:
            case Typedef(old, new):
                return f'typedef {old} {new};'
            case Comment(kind, text):
                return f'// {text}' if kind == 'line' else f'/* {text} */'
            case VarDecl():
                return self._generate_var_decl(stmt)
            case MultiDecl():
                return self._generate_multi_decl(stmt)
            case Block():
                return self._generate_block(stmt)
            case Region():
                return self._generate_region(stmt)
            case If():
                return self._generate_if(stmt)
            case While():
                return self._generate_while(stmt)
            case For():
                return self._generate_for(stmt)
            case DoWhile():
                return self._generate_do_while(stmt)
            case Break():
                return 'break;'
            case Continue():
                return 'continue;'
            case ExprStmt():
                if stmt.expr is None:
                    return ';'
                else:
                    return self._generate_expr(stmt.expr) + ';'
            case _:
                raise ValueError('Unknown statement type: '
                                 f'{type(stmt).__name__}')

    def _generate_init(self, node):
        if isinstance(node, ArrayInit):
            return self._generate_array_init(node)
        else:
            return self._generate_subexpr(node)

    def _generate_sizes(self, sizes):
        return ''.join(f'[{self._generate_expr(s)}]' for s in sizes)

    def _generate_var_decl(self, node, semi=True):
        result = 'const ' * node.const + f'{node.vtype} {node.name}'

        if node.sizes:
            result += self._generate_sizes(node.sizes)

        if node.init:
            result += ' = ' + self._generate_init(node.init)

        return result + ';' if semi else result

    def _generate_multi_decl(self, node, semi=True):
        first = node.decls[0]
        prefix = 'const ' * first.const + first.vtype + ' '

        parts = []
        for d in node.decls:
            part = d.name
            if d.sizes:
                part += self._generate_sizes(d.sizes)
            if d.init:
                part += ' = ' + self._generate_init(d.init)

            parts.append(part)

        result = prefix + ', '.join(parts)
        return result + ';' if semi else result

    def _generate_array_init(self, node):
        parts = [self._generate_init(v) for v in node.values]

        return '{' + ', '.join(parts) + '}'

    def _generate_block(self, node, marker=None):
        self.nind += 1
        indent = self.nind*self.ind
        body = [indent + self._gen_comments(s, indent) +
                self._generate_statement(s) for s in node.body]

        if marker:
            body = [indent + ln for ln in marker.split('\n')] + body

        lines = ['{'] + body
        self.nind -= 1
        lines.append(self.nind*self.ind + '}')

        return '\n'.join(lines)

    def _format_stmt_body(self, stmt, comments=True):
        sstr = self._generate_statement(stmt)

        # Indent an unbraced body one level past its keyword
        if isinstance(stmt, (Block, Region)):
            ind = self.nind*self.ind
        else:
            ind = (self.nind + 1)*self.ind

        pre = self._gen_comments(stmt, ind) if comments else ''
        return f'\n{ind}{pre}{sstr}'

    def _generate_if(self, node):
        indent = self.nind*self.ind
        result = (f'if ({self._generate_expr(node.cond)})'
                  f'{self._format_stmt_body(node.then)}')

        # Emit any comments on the else branch ahead of the keyword
        if isinstance(node.else_, If):
            pre = self._gen_comments(node.else_, indent)
            estr = self._generate_if(node.else_)
            result += f'\n{indent}{pre}else {estr}'
        elif node.else_:
            pre = self._gen_comments(node.else_, indent)
            result += (f'\n{indent}{pre}else'
                       f'{self._format_stmt_body(node.else_, comments=False)}')

        return result

    def _generate_while(self, node):
        return (f'while ({self._generate_expr(node.cond)})'
                + self._format_stmt_body(node.body))

    def _generate_for(self, node):
        if node.init is None:
            istr = ''
        elif isinstance(node.init, VarDecl):
            istr = self._generate_var_decl(node.init, semi=False)
        elif isinstance(node.init, MultiDecl):
            istr = self._generate_multi_decl(node.init, semi=False)
        else:
            istr = self._generate_expr(node.init)

        cstr = self._generate_expr(node.cond) if node.cond else ''
        sstr = self._generate_expr(node.step) if node.step else ''

        return (f'for ({istr}; {cstr}; {sstr})'
                + self._format_stmt_body(node.body))

    def _generate_do_while(self, node):
        bstr = self._generate_statement(node.body)
        cond = self._generate_expr(node.cond)

        if isinstance(node.body, Block):
            return f'do{bstr} while ({cond});'
        else:
            return f'do\n{(self.nind + 1)*self.ind}{bstr} while ({cond});'

    def _get_precedence(self, expr):
        match expr:
            case Binary(op, _, _):
                return PRECEDENCE[op]
            case Ternary(_, _, _):
                return PRECEDENCE['?']
            case Assign(_, _):
                return PRECEDENCE['=']
            case _:
                return self._non_op_prec

    def _generate_assign(self, node):
        lstr = self._generate_expr(node.left)

        # Recover a compound assignment where the target is reused
        match node.right:
            case Binary(op, l, r) if op in self._compound and l == node.left:
                return f'{lstr} {op}= {self._generate_subexpr(r)}'
            case _:
                return f'{lstr} = {self._generate_subexpr(node.right)}'

    def _generate_subexpr(self, expr):
        # Parenthesise comma expressions in single-expression contexts
        estr = self._generate_expr(expr)
        if self._get_precedence(expr) < PRECEDENCE['=']:
            estr = f'({estr})'

        return estr

    def _generate_expr(self, expr):
        match expr:
            case Int(val, suffix, ishex):
                return (hex(val) if ishex else str(val)) + suffix
            case Float(val, suffix, ishex):
                return (val.hex() if ishex else str(val)) + suffix
            case String(val):
                return f'"{val.translate(self._str_esc)}"'
            case Var(name):
                return name
            case Binary(op, left, right):
                prec = PRECEDENCE[op]
                if op == ',':
                    return (f'{self._generate_expr(left)},'
                            f'{self._generate_expr(right)}')
                # ** has no C spelling, so lower it to a call
                elif op == '**':
                    return (f'pow({self._generate_expr(left)}, '
                            f'{self._generate_expr(right)})')
                else:
                    lstr = self._generate_expr(left)
                    rstr = self._generate_expr(right)

                    # Equal precedence needs parens unless right associative
                    rprec = self._get_precedence(right)
                    if rprec < prec + (op not in RIGHT_ASSOC):
                        rstr = f'({rstr})'

                    if self._get_precedence(left) < prec:
                        lstr = f'({lstr})'

                    return f'{lstr} {op} {rstr}'
            case Unary(op, operand):
                ostr = self._generate_expr(operand)
                if isinstance(operand, (Binary, Ternary, Assign, Unary)):
                    ostr = f'({ostr})'

                return f'{op}{ostr}'
            case Postfix(op, operand):
                return f'{self._generate_expr(operand)}{op}'
            case Ternary(cond, true, false):
                tprec = PRECEDENCE['?']

                # A condition of equal precedence would reassociate
                cstr = self._generate_expr(cond)
                if self._get_precedence(cond) <= tprec:
                    cstr = f'({cstr})'

                tstr = self._generate_expr(true)
                if self._get_precedence(true) < tprec:
                    tstr = f'({tstr})'

                fstr = self._generate_expr(false)
                if self._get_precedence(false) < tprec:
                    fstr = f'({fstr})'

                return f'{cstr} ? {tstr} : {fstr}'
            case Assign():
                return self._generate_assign(expr)
            case Call(Var(name), args):
                astr = ', '.join(self._generate_subexpr(a) for a in args)

                return f'{self._func_canon.get(name, name)}({astr})'
            case Call(func, args):
                astr = ', '.join(self._generate_subexpr(a) for a in args)

                return f'{self._generate_expr(func)}({astr})'
            case Index(array, index):
                arr = self._generate_expr(array)
                idx = self._generate_expr(index)

                return f'{arr}[{idx}]'
            case Cast(ctype, inner):
                istr = self._generate_expr(inner)
                if isinstance(inner, (Binary, Ternary, Assign)):
                    istr = f'({istr})'

                return f'({ctype}){istr}'
            case DslCall(name, args):
                raise ValueError(f'Unresolved DSL call: ${name}')
            case DslVar(name):
                raise ValueError(f'Unresolved DSL variable: ${name}')
            case _:
                raise ValueError('Unknown expression type: '
                                 f'{type(expr).__name__}')
