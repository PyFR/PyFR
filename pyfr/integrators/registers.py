class BaseRegister:
    extent = None


class ScalarRegister(BaseRegister):
    vector = False
    dynamic = False

    def __init__(self, *, n=1, rhs=True, extent=None):
        self.n = n
        self.rhs = rhs
        self.extent = extent


class DynamicScalarRegister(BaseRegister):
    vector = False
    dynamic = True

    def __init__(self, *, rhs=True, extent=None):
        self.n = 'dyn'
        self.rhs = rhs
        self.extent = extent


class VectorRegister(BaseRegister):
    vector = True
    dynamic = False

    def __init__(self, *, n, rhs=True, extent=None):
        self.n = n
        self.rhs = rhs
        self.extent = extent


class DynamicVectorRegister(BaseRegister):
    vector = True
    dynamic = True

    def __init__(self, *, rhs=True, extent=None):
        self.n = 'dyn'
        self.rhs = rhs
        self.extent = extent


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
