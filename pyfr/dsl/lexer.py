from dataclasses import dataclass
import re

from pyfr.dsl.types import SCALAR_TYPES


_KEYWORDS = 'if else while for const break continue do typedef'.split()
_TYPES = frozenset(SCALAR_TYPES) | {'void'}
_OPS = '<<= >>= == != <= >= && || ++ -- += -= *= /= %= &= |= ^= << >> **'
_ESC = dict(zip('ntr\\"\'', '\n\t\r\\"\''))

# Hex and base 10 literals (integer or float)
_HEX = r'0[xX][\da-fA-F]*(?:\.[\da-fA-F]*)?(?:[pP][+-]?\d*)?'
_DEC = r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d*)?'

_TOKEN = re.compile('(?:' + '|'.join([
    # Line comment (//)
    r'//(?P<line>[^\n]*)',
    # Block comment (/*)
    r'/\*(?P<block>.*?)\*/',
    # Directive, of which we only support #pragma
    r'\#[ \t]*pragma(?P<PRAGMA>[^\n]*)',
    # Identifier/keyword or @-prefixed DSL identifier
    r'@(?P<DSL>[A-Za-z_]\w*)',
    r'(?P<IDENT>[A-Za-z_]\w*)',
    # Number
    rf'(?P<NUM>(?P<RAW>{_HEX}|{_DEC})(?P<SUF>[uUlLfF]*))',
    # String literal
    r'''(?P<STRING>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''',
    # Unterminated comment or string, bare @ or unknown directive
    r'''(?P<BAD>/\*|["'@#])''',
    # Symbol (check longest operators first)
    rf'(?P<OP>{"|".join(map(re.escape, _OPS.split()))}|\S)'
]) + r')\s*', re.A | re.S)


@dataclass(slots=True)
class Token:
    kind: str
    value: object
    pos: int


class Lexer:
    def __init__(self, src, types=()):
        self.src = src

        # Kinds of the reserved words, with keywords being their own kind
        self.kinds = dict(zip(_KEYWORDS, _KEYWORDS))
        self.kinds |= dict.fromkeys(_TYPES | set(types), 'TYPE')

    def __iter__(self):
        kinds = self.kinds

        for m in _TOKEN.finditer(self.src):
            kind = m.lastgroup
            text, pos = m[kind], m.start()

            match kind:
                case 'OP':
                    yield Token(text, text, pos)
                case 'IDENT':
                    yield Token(kinds.get(text, kind), text, pos)
                case 'NUM':
                    yield Token(*self._number(m['RAW'], m['SUF'], pos), pos)
                case 'STRING':
                    yield Token(kind, self._unescape(text[1:-1]), pos)
                case 'line' | 'block':
                    yield Token('COMMENT', (kind, text.strip()), pos)
                case 'BAD':
                    raise self.error(f'Invalid {text!r}', pos)
                case _:
                    yield Token(kind, text.strip(), pos)

        yield Token('EOF', '', len(self.src))

    def error(self, msg, pos):
        src = self.src
        line, col = src.count('\n', 0, pos) + 1, pos - src.rfind('\n', 0, pos)

        return SyntaxError(f'{msg} at {line}:{col}')

    def _number(self, raw, suffix, pos):
        ishex = raw[:2] in ('0x', '0X')

        # Take whatever does not read as an integer to be a float
        try:
            value = int(raw, 16 if ishex else 10)
        except ValueError:
            try:
                value = float.fromhex(raw) if ishex else float(raw)
            except ValueError:
                raise self.error(f'Invalid number {raw!r}', pos) from None

        # An f suffix makes a float of what parsed as an integer
        isfloat = isinstance(value, float) or suffix in ('f', 'F')

        return 'FLOAT' if isfloat else 'INT', (value, suffix, ishex)

    def _unescape(self, text):
        return re.sub(r'\\(.)', lambda m: _ESC.get(m[1], '\\'), text,
                      flags=re.S)
