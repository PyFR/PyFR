# -*- coding: utf-8 -*-

import itertools as it
import re
import types

from pyfr.util import memoize


class Kernel:
    def __init__(self, mats=[], views=[], misc=[]):
        self.mats = mats
        self.views = views
        self.misc = misc

    @property
    def retval(self):
        return None

    def run(self, queue, *args, **kwargs):
        pass


class NullKernel(Kernel):
    pass


class MetaKernel(Kernel):
    def __init__(self, kernels):
        self._kernels = list(kernels)

    def run(self, queue, *args, **kwargs):
        for k in self._kernels:
            k.run(queue, *args, **kwargs)


class BaseKernelProvider:
    def __init__(self, backend):
        self.backend = backend


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
            raise ValueError(f'Kernel "{name}" not defined in template')

        # Extract the metadata for the kernel
        ndim, argn, argt = argspecs[name]

        return src, ndim, argn, argt

    def _build_kernel(self, name, src, args):
        pass

    def _build_arglst(self, dims, argn, argt, argdict):
        # Possible matrix types
        mattypes = (
            self.backend.const_matrix_cls, self.backend.matrix_cls,
            self.backend.xchg_matrix_cls, self.backend.matrix_slice_cls
        )

        # Possible view types
        viewtypes = (self.backend.view_cls, self.backend.xchg_view_cls)

        # Backend matrices and views this kernel operates on
        argmats, argviews = [], []

        # First arguments are the iteration dimensions
        ndim, arglst = len(dims), [int(d) for d in dims]

        # Followed by the objects themselves
        for aname, atypes in zip(argn[ndim:], argt[ndim:]):
            try:
                ka = argdict[aname]
            except KeyError:
                # Allow scalar arguments to be resolved at runtime
                if len(atypes) == 1 and atypes[0] == self.backend.fpdtype:
                    ka = aname
                else:
                    raise

            # Matrix
            if isinstance(ka, mattypes):
                argmats.append(ka)

                # Check that argument is not a row sliced matrix
                if isinstance(ka, mattypes[-1]) and ka.nrow != ka.parent.nrow:
                    raise ValueError('Row sliced matrices are not supported')
                else:
                    arglst += [ka, ka.leaddim] if len(atypes) == 2 else [ka]
            # View
            elif isinstance(ka, viewtypes):
                argviews.append(ka)

                if isinstance(ka, self.backend.view_cls):
                    view = ka
                else:
                    view = ka.view

                arglst += [view.basedata, view.mapping]
                arglst += [view.rstrides] if len(atypes) == 3 else []
            # Other; let the backend handle it
            else:
                arglst.append(ka)

        return arglst, (argmats, argviews)

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        pass

    def register(self, mod):
        # Derive the name of the kernel from the module
        name = mod[mod.rfind('.') + 1:]

        # See if a kernel has already been registered under this name
        if hasattr(self, name):
            # Same name different module
            if getattr(self, name)._mod != mod:
                raise RuntimeError(f'Attempt to re-register "{name}" with a '
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
            fun = self._build_kernel(name, src, list(it.chain(*argt)))

            # Process the argument list
            argb, argmv = self._build_arglst(dims, argn, argt, kwargs)

            # Return a Kernel subclass instance
            return self._instantiate_kernel(dims, fun, argb, argmv)

        # Attach the module to the method as an attribute
        kernel_meth._mod = mod

        # Bind
        setattr(self, name, types.MethodType(kernel_meth, self))


class NotSuitableError(Exception):
    pass
