from dataclasses import dataclass


_KEYWORDS = frozenset(
    'if else while for const break continue do typedef'.split()
)
TYPE_KEYWORDS = frozenset('int float double void'.split())
_STDINT_TYPES = frozenset(
    'int8_t int16_t int32_t int64_t uint8_t uint16_t '
    'uint32_t uint64_t size_t ssize_t fpdtype_t ixdtype_t'.split()
)
_TWO_CHAR_OPS = frozenset(
    '== != <= >= && || ++ -- += -= *= /= %= &= |= ^= << >> **'.split()
)
_THREE_CHAR_OPS = frozenset(['<<=', '>>='])

# Lookup tables for lexer
_OPS = {3: {op.encode() for op in _THREE_CHAR_OPS},
        2: {op.encode() for op in _TWO_CHAR_OPS}}
_S = bytearray(c in b' \t\n\r' for c in range(256))
_A = bytearray(chr(c).isalpha() or c == ord('_') for c in range(256))
_D = bytearray(chr(c).isdigit() for c in range(256))
_H = bytearray(chr(c) in '0123456789abcdefABCDEF' for c in range(256))
_I = bytearray(a or d for a, d in zip(_A, _D))
_ESC = dict(zip(map(ord, 'ntr\\"\''), '\n\t\r\\"\''))

# Character constants
_NL = ord('\n')
_HASH = ord('#')
_DOT = ord('.')
_ZERO = ord('0')
_X, _x = ord('X'), ord('x')
_E, _e = ord('E'), ord('e')
_P, _p = ord('P'), ord('p')
_PLUS, _MINUS = ord('+'), ord('-')
_DQUOTE, _SQUOTE = ord('"'), ord("'")
_BACKSLASH = ord('\\')
_SLASH, _STAR = ord('/'), ord('*')
_DOLLAR = ord('$')
_SUFFIXES = frozenset(map(ord, 'uUlLfF'))


@dataclass(slots=True)
class Token:
    kind: str
    value: object
    line: int
    col: int
    comments: tuple = ()


class Lexer:
    def __init__(self, src, types=()):
        self.src = src.encode() if isinstance(src, str) else src
        self.types = TYPE_KEYWORDS | _STDINT_TYPES | set(types)
        self.keywords = _KEYWORDS

    def tokenise(self):
        src, types, n = self.src, self.types, len(self.src)
        keywords = self.keywords
        i, line, col = 0, 1, 1
        tokens, pend = [], []

        # Attach any pending comment trivia to the token being emitted
        def tok(k, v, l, c):
            tokens.append(Token(k, v, l, c, tuple(pend)))
            pend.clear()

        while i < n:
            ch = src[i]

            # Whitespace
            if _S[ch]:
                if ch == _NL:
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
                continue

            sc = col

            # Directive, of which we only support #pragma
            if ch == _HASH:
                j = src.find(b'\n', i)
                j = n if j == -1 else j

                text = src[i + 1:j].decode().strip()
                if not text.startswith('pragma'):
                    raise SyntaxError(f'Unknown directive at {line}:{col}')

                tok('PRAGMA', text[6:].strip(), line, sc)
                col += j - i
                i = j
                continue

            # Identifier/keyword or $-prefixed DSL identifier
            if _A[ch] or ch == _DOLLAR:
                dsl = ch == _DOLLAR
                j = i + dsl
                if j < n and _A[src[j]]:
                    j += 1
                    while j < n and _I[src[j]]:
                        j += 1
                elif dsl:
                    raise SyntaxError(f'Bare $ at {line}:{col}')

                ident = src[i + dsl:j].decode()
                if dsl:
                    kind = 'DSL'
                elif ident in types:
                    kind = 'TYPE'
                elif ident in keywords:
                    kind = 'KEYWORD'
                else:
                    kind = 'IDENT'

                tok(kind, ident, line, sc)
                col += j - i
                i = j
                continue

            # Number
            if _D[ch] or (ch == _DOT and i + 1 < n and _D[src[i + 1]]):
                kind, value, j = self._lex_number(i, line, col)
                tok(kind, value, line, sc)
                col += j - i
                i = j
                continue

            # String literal
            if ch == _DQUOTE or ch == _SQUOTE:
                val, j = self._lex_string(i, line, col)
                tok('STRING', val, line, sc)
                col += j - i
                i = j
                continue

            # Comment
            if ch == _SLASH and i + 1 < n:
                nc = src[i + 1]

                # Line comment (//)
                if nc == _SLASH:
                    j = i + 2
                    while j < n and src[j] != _NL:
                        j += 1

                    content = src[i + 2:j].decode().strip()
                    pend.append(('line', content))
                    col += j - i
                    i = j
                    continue

                # Block comment (/*)
                if nc == _STAR:
                    j = i + 2
                    col += 2
                    while j < n - 1:
                        # End of block comment (*/)
                        if src[j] == _STAR and src[j + 1] == _SLASH:
                            content = src[i + 2:j].decode().strip()
                            pend.append(('block', content))
                            col += 2
                            j += 2
                            break

                        # Track newlines for line/col
                        if src[j] == _NL:
                            line += 1
                            col = 1
                        else:
                            col += 1

                        j += 1
                    else:
                        raise SyntaxError('Unterminated comment at '
                                          f'{line}:{col}')

                    i = j
                    continue

            # Symbol (check longest operators first)
            for length in (3, 2):
                if i + length <= n and src[i:i + length] in _OPS[length]:
                    tok('SYMBOL', src[i:i + length].decode(), line, sc)
                    i += length
                    col += length
                    break
            else:
                tok('SYMBOL', chr(ch), line, sc)
                i += 1
                col += 1

        tok('EOF', '', line, col)
        return tokens

    def _lex_number(self, i, line, col):
        src, n = self.src, len(self.src)
        j, isfloat, ishex = i, False, False

        # Hex literal (integer or float)
        if src[i] == _ZERO and i + 1 < n and src[i + 1] in (_x, _X):
            j = i + 2
            while j < n and _H[src[j]]:
                j += 1

            # Hex float: fractional part
            if j < n and src[j] == _DOT:
                isfloat, j = True, j + 1
                while j < n and _H[src[j]]:
                    j += 1

            # Hex float: p/P exponent (required for hex floats)
            if j < n and src[j] in (_p, _P):
                isfloat, j = True, j + 1
                if j < n and src[j] in (_PLUS, _MINUS):
                    j += 1

                if j >= n or not _D[src[j]]:
                    raise SyntaxError(f'Invalid hex float at {line}:{col}')

                while j < n and _D[src[j]]:
                    j += 1

            if j == i + 2:
                raise SyntaxError(f'Invalid hex at {line}:{col}')

            raw = src[i:j].decode()
            value = float.fromhex(raw) if isfloat else int(raw, 16)
            ishex = True
        # Otherwise base 10
        else:
            # Integer part
            while j < n and _D[src[j]]:
                j += 1

            # Decimal part
            if j < n and src[j] == _DOT:
                isfloat, j = True, j + 1
                while j < n and _D[src[j]]:
                    j += 1

            # Exponent
            if j < n and src[j] in (_e, _E):
                isfloat, j = True, j + 1
                if j < n and src[j] in (_PLUS, _MINUS):
                    j += 1

                if j >= n or not _D[src[j]]:
                    raise SyntaxError(f'Invalid float at {line}:{col}')

                while j < n and _D[src[j]]:
                    j += 1

            num = src[i:j].decode()
            value = float(num) if isfloat else int(num)

        # Suffix
        k = j
        while j < n and src[j] in _SUFFIXES:
            j += 1

        suffix = src[k:j].decode()

        # An f suffix makes a float of what parsed as an integer
        isfloat |= suffix in ('f', 'F')

        return 'FLOAT' if isfloat else 'INT', (value, suffix, ishex), j

    def _lex_string(self, i, line, col):
        src, n = self.src, len(self.src)
        quote, j, val = src[i], i + 1, ''

        while j < n:
            c = src[j]

            # Closing quote
            if c == quote:
                j += 1
                break

            # Escape sequence
            if c == _BACKSLASH:
                j += 1
                if j >= n:
                    raise SyntaxError(f'Unterminated escape at {line}:{col}')

                val += _ESC.get(src[j], '\\')
                j += 1
            # Regular character
            else:
                val += chr(c)
                j += 1
        else:
            raise SyntaxError(f'Unterminated string at {line}:{col}')

        return val, j


class PlainLexer(Lexer):
    def __init__(self, src):
        super().__init__(src)

        # Treat every word as an identifier
        self.types = self.keywords = frozenset()
