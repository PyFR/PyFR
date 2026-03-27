class BaseRegister:
    pass


class ScalarRegister(BaseRegister):
    vector = False
    dynamic = False

    def __init__(self, *, n=1, rhs=True):
        self.n = n
        self.rhs = rhs


class DynamicScalarRegister(BaseRegister):
    vector = False
    dynamic = True

    def __init__(self, *, rhs=True):
        self.n = 'dyn'
        self.rhs = rhs


class VectorRegister(BaseRegister):
    vector = True
    dynamic = False

    def __init__(self, *, n, rhs=True):
        self.n = n
        self.rhs = rhs


class DynamicVectorRegister(BaseRegister):
    vector = True
    dynamic = True

    def __init__(self, *, rhs=True):
        self.n = 'dyn'
        self.rhs = rhs


class RegisterMeta(type):
    def __new__(mcls, name, bases, namespace):
        registers = {}
        for base in reversed(bases):
            registers |= getattr(base, '_registers', {})

        for k, v in namespace.items():
            if isinstance(v, BaseRegister):
                registers[v] = k

        namespace['_registers'] = registers
        return super().__new__(mcls, name, bases, namespace)
