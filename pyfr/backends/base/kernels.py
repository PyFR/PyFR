# -*- coding: utf-8 -*-

import itertools as it
import types

from pyfr.util import memoize, proxylist


class _BaseKernel(object):
    def __call__(self, *args, **kwargs):
        return self, args, kwargs

    @property
    def retval(self):
        return None

    def run(self, queue, *args, **kwargs):
        pass


class ComputeKernel(_BaseKernel):
    ktype = 'compute'


class MPIKernel(_BaseKernel):
    ktype = 'mpi'


class NullComputeKernel(ComputeKernel):
    pass


class NullMPIKernel(MPIKernel):
    pass


class _MetaKernel(object):
    def __init__(self, kernels):
        self._kernels = proxylist(kernels)

    def run(self, queue, *args, **kwargs):
        self._kernels.run(queue, *args, **kwargs)


class ComputeMetaKernel(_MetaKernel, ComputeKernel):
    pass


class MPIMetaKernel(_MetaKernel, MPIKernel):
    pass


class BaseKernelProvider(object):
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

        # Check the kernel exists in the template
        if name not in argspecs:
            raise ValueError('Kernel "{0}" not defined in template'
                             .format(name))

        # Extract the metadata for the kernel
        ndim, argn, argt = argspecs[name]

        return src, ndim, argn, argt

    def _build_kernel(self, name, src, args):
        pass

    def _build_arglst(self, dims, argn, argt, argdict):
        # Possible matrix types
        mattypes = (
            self.backend.const_matrix_cls, self.backend.matrix_cls,
            self.backend.matrix_bank_cls, self.backend.matrix_slice_cls,
            self.backend.xchg_matrix_cls
        )

        # Possible view types
        viewtypes = (self.backend.view_cls, self.backend.xchg_view_cls)

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
                arglst += [ka, ka.leaddim] if len(atypes) == 2 else [ka]
            # View
            elif isinstance(ka, viewtypes):
                if isinstance(ka, self.backend.view_cls):
                    view = ka
                else:
                    view = ka.view

                arglst += [view.basedata, view.mapping]
                arglst += [view.rstrides] if len(atypes) == 3 else []
            # Other; let the backend handle it
            else:
                arglst.append(ka)

        return arglst

    def _instantiate_kernel(self, dims, fun, arglst):
        pass

    def register(self, mod):
        # Derive the name of the kernel from the module
        name = mod[mod.rfind('.') + 1:]

        # See if a kernel has already been registered under this name
        if hasattr(self, name):
            # Same name different module
            if getattr(self, name)._mod != mod:
                raise RuntimeError('Attempt to re-register "{0}" with a '
                                   'different module'.format(name))
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
            argb = self._build_arglst(dims, argn, argt, kwargs)

            # Return a ComputeKernel subclass instance
            return self._instantiate_kernel(dims, fun, argb)

        # Attach the module to the method as an attribute
        kernel_meth._mod = mod

        # Bind
        setattr(self, name, types.MethodType(kernel_meth, self))


class NotSuitableError(Exception):
    pass
