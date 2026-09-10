from contextlib import contextmanager
from functools import cached_property
import re
import types

import numpy as np

from pyfr.cache import memoize


class Kernel:
    compound = False

    def __init__(self, args={}, mats=[], misc=[], dt=float('nan')):
        self.args = args
        self.mats = mats
        self.misc = misc
        self.dt = dt

        # Arguments sealed by committed graph group substitutions
        self.sealed = set()

    @property
    def retval(self):
        return None

    @property
    def leaves(self):
        return [self]

    @cached_property
    def argnames(self):
        return frozenset(self.args)

    def bind(self, **kwargs):
        # Rebind the named arguments
        for n, v in kwargs.items():
            i, spec, curr = self.args[n]
            form = spec[0]

            if i in self.sealed:
                raise RuntimeError(f'Argument {n} is sealed by a graph '
                                   'substitution')

            # Validate the argument and expand it into kernel arguments
            if form == 's' and isinstance(v, int | float):
                kargs = [v]
            elif form == 's' or isinstance(v, int | float):
                raise ValueError('Rebound argument must be of identical kind')
            elif v.traits == curr.traits:
                kargs = v.kargs(form)
            else:
                raise ValueError(f'Trait mismatch {v.traits} != {curr.traits}')

            # Update the underlying kernel
            for j, s in enumerate(kargs, start=i):
                self._set_arg(j, s)

            self.args[n] = (i, spec, v)

    def _set_arg(self, i, v):
        pass

    def run(self, *args):
        pass


class NullKernel(Kernel):
    pass


class BaseMetaKernel(Kernel):
    def __init__(self, kernels):
        super().__init__()

        self.kernels = list(kernels)

    @property
    def leaves(self):
        return [l for k in self.kernels for l in k.leaves]

    def run(self, *args):
        for k in self.kernels:
            k.run(*args)


class BaseOrderedMetaKernel(BaseMetaKernel):
    pass


class BaseUnorderedMetaKernel(BaseMetaKernel):
    def __init__(self, kernels, splits):
        super().__init__(kernels)

        if splits is not None:
            self.splits = list(splits)
            self.compound = True

            if len(self.splits) != len(self.kernels) - 1:
                raise ValueError('Invalid split points')


class BaseKernelProvider:
    def __init__(self, backend):
        self.backend = backend

    def _bench_rand_block(self, nbytes):
        # Generate up to 64 MiB of random data
        n = min(nbytes, 2**26)
        blk = np.random.default_rng(0).integers(0, 256, n, dtype=np.uint8)

        # Ensure that the data can be interpreted as floating point values
        blk[(blk & 0x7c) == 0x7c] &= 0x7b

        return blk

    @contextmanager
    def _bench_data(self, save=[], rand=[]):
        self.backend.wait()

        # Resolve any slices to their parent matrices
        mats = [getattr(m, 'parent', m) for m in [*save, *rand]]

        # Save the operands
        saved = [(m, self._bench_save(m)) for m in mats]

        # Fill with random entries
        for m in mats[len(save):]:
            self._bench_fill_random(m)

        try:
            yield
        finally:
            # Restore the saved operands
            for m, buf in saved:
                self._bench_restore(m, buf)


class BasePointwiseKernelProvider(BaseKernelProvider):
    kernel_generator_cls = None

    @memoize
    def _render_kernel(self, name, mod, extrns, tplargs):
        # Copy the provided argument list
        tplargs = dict(tplargs)

        # Backend-specfic generator classes
        tplargs['_kernel_generator'] = self.kernel_generator_cls

        # Macro definitions
        tplargs['_macros'] = {}

        # External kernel arguments dictionary
        tplargs['_extrns'] = extrns

        # Backchannel for obtaining kernel argument types
        tplargs['_kernel_argspecs'] = argspecs = {}

        # Render the template to yield the source code
        tpl = self.backend.lookup.get_template(mod)
        src = tpl.render(**tplargs)
        src = re.sub(r'\n\n+', r'\n\n', src)

        # Check the kernel exists in the template
        if name not in argspecs:
            raise ValueError(f'Kernel {name!r} not defined in template')

        # Extract the metadata for the kernel
        ndim, argn, argt = argspecs[name]

        return src, ndim, argn, argt

    def _build_kernel(self, name, src, args):
        pass

    def _build_args(self, ndim, argn, argt, argdict):
        # Named arguments along with their indices and specs
        args, i = {}, ndim

        # Arguments are laid out after the iteration dimensions
        for aname, (form, atypes) in zip(argn[ndim:], argt[ndim:]):
            # Default unbound scalar arguments to zero
            if form == 's':
                if atypes[0] == self.backend.fpdtype:
                    ka = float(argdict.get(aname, 0))
                else:
                    ka = int(argdict.get(aname, 0))
            else:
                ka = argdict[aname]
                mscls = self.backend.matrix_slice_cls

                # Check that argument is not a row sliced matrix
                if isinstance(ka, mscls) and ka.nrow != ka.parent.nrow:
                    raise ValueError('Row sliced matrices are not supported')

            args[aname] = (i, (form, atypes), ka)
            i += len(atypes)

        return args

    def _instantiate_kernel(self, dims, fun, args):
        pass

    def register(self, mod):
        # Derive the name of the kernel from the module
        name = mod[mod.rfind('.') + 1:]

        # See if a kernel has already been registered under this name
        if hasattr(self, name):
            # Same name different module
            if getattr(self, name)._mod != mod:
                raise RuntimeError(f'Attempt to re-register {name!r} with a '
                                   'different module')
            # Otherwise (since we're already registered) return
            else:
                return

        # Generate the kernel providing method
        def kernel_meth(self, tplargs, dims, extrns={}, **kwargs):
            # Render the source of kernel
            src, ndim, argn, argt = self._render_kernel(name, mod, extrns,
                                                        tplargs)

            # Compile the kernel
            argtypes = [t for _, ts in argt for t in ts]
            fun = self._build_kernel(name, src, argtypes)

            # Process the argument list
            args = self._build_args(len(dims), argn, argt, kwargs)

            # Create the kernel and set its arguments
            kern = self._instantiate_kernel(dims, fun, args)
            kern.bind(**{n: v for n, (_, _, v) in args.items()})

            return kern

        # Attach the module to the method as an attribute
        kernel_meth._mod = mod

        # Bind
        setattr(self, name, types.MethodType(kernel_meth, self))


class NotSuitableError(Exception):
    pass
