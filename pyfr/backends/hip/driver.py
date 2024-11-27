from ctypes import (POINTER, Structure, addressof, byref, cast,
                    create_string_buffer, c_char, c_char_p, c_float, c_int,
                    c_size_t, c_uint, c_void_p)
from uuid import UUID

import numpy as np

from pyfr.ctypesutil import LibWrapper


# Possible HIP exception types
class HIPError(Exception): pass
class HIPInvalidValue(HIPError): pass
class HIPNotInitialized(HIPError): pass
class HIPOutOfMemory(HIPError): pass
class HIPInsufficientDriver(HIPError): pass
class HIPPriorLaunchFailure(HIPError): pass
class HIPInvalidDevice(HIPError): pass
class HIPECCNotCorrectable(HIPError): pass
class HIPFileNotFound(HIPError): pass
class HIPNotFound(HIPError): pass
class HIPIllegalAddress(HIPError): pass
class HIPLaunchFailure(HIPError): pass


class HIPKernelNodeParams(Structure):
    _fields_ = [
        ('block', c_uint * 3),
        ('extra', POINTER(c_void_p)),
        ('func', c_void_p),
        ('grid', c_uint * 3),
        ('kernel_params', POINTER(c_void_p)),
        ('shared_mem_bytes', c_uint)
    ]

    def __init__(self, func, grid, block, sharedb):
        super().__init__()

        # Save a reference to our underlying function
        self._func = func

        # For each argument type instantiate the corresponding ctypes type
        self._args = args = [atype() for atype in func.argtypes]

        # Obtain pointers to these arguments
        self._arg_ptrs = (c_void_p * len(args))(*map(addressof, args))

        # Fill out the structure
        self.func = int(func)
        self.grid[:] = grid
        self.block[:] = block
        self.shared_mem_bytes = sharedb
        self.kernel_params = self._arg_ptrs

    def set_arg(self, i, v):
        self._args[i].value = getattr(v, '_as_parameter_', v)

    def set_args(self, *kargs, start=0):
        for i, v in enumerate(kargs, start=start):
            self.set_arg(i, v)


class HIPWrappers(LibWrapper):
    _libname = 'amdhip64'

    # Error codes
    _statuses = {
        1: HIPInvalidValue,
        2: HIPOutOfMemory,
        3: HIPNotInitialized,
        35: HIPInsufficientDriver,
        53: HIPPriorLaunchFailure,
        101: HIPInvalidDevice,
        214: HIPECCNotCorrectable,
        301: HIPFileNotFound,
        500: HIPNotFound,
        700: HIPIllegalAddress,
        719: HIPLaunchFailure,
        '*': HIPError
    }

    # Constants
    FUNC_ATTR_SHARED_SIZE_BYTES = 1
    FUNC_ATTR_LOCAL_SIZE_BYTES = 3
    FUNC_ATTR_NUM_REGS = 4
    MEMCPY_DEFAULT = 4

    # Functions
    _functions = [
        (c_int, 'hipRuntimeGetVersion', POINTER(c_int)),
        (c_int, 'hipGetDeviceCount', POINTER(c_int)),
        (c_int, 'hipGetDevicePropertiesR0600', c_void_p, c_int),
        (c_int, 'hipSetDevice', c_int),
        (c_int, 'hipDeviceGetUuid', 16*c_char, c_int),
        (c_int, 'hipMalloc', POINTER(c_void_p), c_size_t),
        (c_int, 'hipFree', c_void_p),
        (c_int, 'hipHostMalloc', POINTER(c_void_p), c_size_t, c_uint),
        (c_int, 'hipHostFree', c_void_p),
        (c_int, 'hipMemcpy', c_void_p, c_void_p, c_size_t, c_int),
        (c_int, 'hipMemcpyAsync', c_void_p, c_void_p, c_size_t, c_int,
         c_void_p),
        (c_int, 'hipMemset', c_void_p, c_int, c_size_t),
        (c_int, 'hipStreamCreate', POINTER(c_void_p)),
        (c_int, 'hipStreamDestroy', c_void_p),
        (c_int, 'hipStreamBeginCapture', c_void_p, c_uint),
        (c_int, 'hipStreamEndCapture', c_void_p, POINTER(c_void_p)),
        (c_int, 'hipStreamSynchronize', c_void_p),
        (c_int, 'hipEventCreate', POINTER(c_void_p)),
        (c_int, 'hipEventRecord', c_void_p, c_void_p),
        (c_int, 'hipEventDestroy', c_void_p),
        (c_int, 'hipEventSynchronize', c_void_p),
        (c_int, 'hipEventElapsedTime', POINTER(c_float), c_void_p, c_void_p),
        (c_int, 'hipModuleLoadData', POINTER(c_void_p), c_char_p),
        (c_int, 'hipModuleUnload', c_void_p),
        (c_int, 'hipModuleGetFunction', POINTER(c_void_p), c_void_p, c_char_p),
        (c_int, 'hipModuleLaunchKernel', c_void_p, c_uint, c_uint, c_uint,
         c_uint, c_uint, c_uint, c_uint, c_void_p, POINTER(c_void_p),
         c_void_p),
        (c_int, 'hipFuncGetAttribute', POINTER(c_int), c_int, c_void_p),
        (c_int, 'hipGraphCreate', POINTER(c_void_p), c_uint),
        (c_int, 'hipGraphDestroy', c_void_p),
        (c_int, 'hipGraphAddEmptyNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t),
        (c_int, 'hipGraphAddEventRecordNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p),
        (c_int, 'hipGraphAddKernelNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, POINTER(HIPKernelNodeParams)),
        (c_int, 'hipGraphAddChildGraphNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p),
        (c_int, 'hipGraphAddMemcpyNode1D', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p, c_void_p, c_size_t, c_int),
        (c_int, 'hipGraphInstantiate', POINTER(c_void_p), c_void_p, c_void_p,
         c_char_p, c_size_t),
        (c_int, 'hipGraphExecKernelNodeSetParams', c_void_p, c_void_p,
         POINTER(HIPKernelNodeParams)),
        (c_int, 'hipGraphExecDestroy', c_void_p),
        (c_int, 'hipGraphLaunch', c_void_p, c_void_p)
    ]

    def _transname(self, name):
        return name.removesuffix('R0600')


class _HIPBase:
    _destroyfn = None

    def __init__(self, hip, ptr):
        self.hip = hip
        self._as_parameter_ = ptr.value

    def __del__(self):
        if self._destroyfn:
            try:
                getattr(self.hip.lib, self._destroyfn)(self)
            except AttributeError:
                pass

    def __int__(self):
        return self._as_parameter_


class HIPDevAlloc(_HIPBase):
    _destroyfn = 'hipFree'

    def __init__(self, hip, nbytes):
        self.nbytes = nbytes

        ptr = c_void_p()
        hip.lib.hipMalloc(ptr, nbytes)

        super().__init__(hip, ptr)


class HIPHostAlloc(_HIPBase):
    _destroyfn = 'hipHostFree'

    def __init__(self, hip, nbytes):
        self.nbytes = nbytes

        ptr = c_void_p()
        hip.lib.hipHostMalloc(ptr, nbytes, 0)

        super().__init__(hip, ptr)


class HIPStream(_HIPBase):
    _destroyfn = 'hipStreamDestroy'

    def __init__(self, hip):
        ptr = c_void_p()
        hip.lib.hipStreamCreate(ptr)

        super().__init__(hip, ptr)

    def begin_capture(self):
        self.hip.lib.hipStreamBeginCapture(self, 0)

    def end_capture(self):
        graph = c_void_p()
        self.hip.lib.hipStreamEndCapture(self, graph)

        return HIPGraph(self.hip, graph)

    def synchronize(self):
        self.hip.lib.hipStreamSynchronize(self)


class HIPEvent(_HIPBase):
    _destroyfn = 'hipEventDestroy'

    def __init__(self, hip):
        ptr = c_void_p()
        hip.lib.hipEventCreate(ptr)

        super().__init__(hip, ptr)

    def record(self, stream):
        self.hip.lib.hipEventRecord(self, stream)

    def synchronize(self):
        self.hip.lib.hipEventSynchronize(self)

    def elapsed_time(self, start):
        dt = c_float()
        self.hip.lib.hipEventElapsedTime(dt, start, self)

        return dt.value / 1e3


class HIPModule(_HIPBase):
    _destroyfn = 'hipModuleUnload'

    def __init__(self, hip, code):
        ptr = c_void_p()
        hip.lib.hipModuleLoadData(ptr, code)

        super().__init__(hip, ptr)

    def get_function(self, name, argspec):
        return HIPFunction(self.hip, self, name, argspec)


class HIPFunction(_HIPBase):
    def __init__(self, hip, module, name, argtypes):
        ptr = c_void_p()
        hip.lib.hipModuleGetFunction(ptr, module, name.encode())

        super().__init__(hip, ptr)

        self.nreg = self._get_attr('num_regs')
        self.shared_mem = self._get_attr('shared_size_bytes')
        self.local_mem = self._get_attr('local_size_bytes')

        # Save a reference to our underlying module and argument types
        self.module = module
        self.argtypes = list(argtypes)

    def _get_attr(self, attr):
        attr = getattr(self.hip.lib, f'FUNC_ATTR_{attr.upper()}')

        v = c_int()
        self.hip.lib.hipFuncGetAttribute(byref(v), attr, self)

        return v.value

    def make_params(self, grid, block, sharedb=0):
        return HIPKernelNodeParams(self, grid, block, sharedb)

    def exec_async(self, stream, params):
        self.hip.lib.hipModuleLaunchKernel(self, *params.grid, *params.block,
                                           params.shared_mem_bytes, stream,
                                           params.kernel_params, None)


class HIPGraph(_HIPBase):
    _destroyfn = 'hipGraphDestroy'

    def __init__(self, hip, ptr=None):
        if ptr is None:
            ptr = c_void_p()
            hip.lib.hipGraphCreate(ptr, 0)

        super().__init__(hip, ptr)

    @staticmethod
    def _make_deps(deps):
        if deps:
            return (c_void_p * len(deps))(*deps), len(deps)
        else:
            return None, 0

    def add_empty(self, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddEmptyNode(ptr, self, *self._make_deps(deps))

        return ptr.value

    def add_event_record(self, event, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddEventRecordNode(ptr, self,
                                                *self._make_deps(deps), event)

        return ptr.value

    def add_kernel(self, kparams, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddKernelNode(ptr, self, *self._make_deps(deps),
                                           kparams)

        return ptr.value

    def add_memcpy(self, dst, src, nbytes, deps=None):
        kind = self.hip.lib.MEMCPY_DEFAULT

        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        ptr = c_void_p()
        self.hip.lib.hipGraphAddMemcpyNode1D(ptr, self, *self._make_deps(deps),
                                             dst, src, nbytes, kind)

        return ptr.value

    def add_graph(self, graph, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddChildGraphNode(ptr, self,
                                               *self._make_deps(deps), graph)

        return ptr.value

    def instantiate(self):
        return HIPExecGraph(self.hip, self)


class HIPExecGraph(_HIPBase):
    _destroyfn = 'hipGraphExecDestroy'

    def __init__(self, hip, graph):
        ptr = c_void_p()
        hip.lib.hipGraphInstantiate(ptr, graph, None, None, 0)

        super().__init__(hip, ptr)

        # Save a reference to the graph
        self.graph = graph

    def set_kernel_node_params(self, node, kparams):
        self.hip.lib.hipGraphExecKernelNodeSetParams(self, node, kparams)

    def launch(self, stream):
        self.hip.lib.hipGraphLaunch(self, stream)


class HIP:
    def __init__(self):
        self.lib = HIPWrappers()

    def device_count(self):
        count = c_int()
        self.lib.hipGetDeviceCount(count)

        return count.value

    def device_properties(self, devid):
        buf = create_string_buffer(2048)
        self.lib.hipGetDeviceProperties(buf, devid)

        return {
            'gcn_arch_name': cast(buf[1160:], c_char_p).value.decode(),
            'warp_size': cast(buf[308:], POINTER(c_int)).contents.value
        }

    def device_uuid(self, devid):
        buf = create_string_buffer(16)
        self.lib.hipDeviceGetUuid(buf, devid)

        return UUID(bytes=buf.raw)

    def set_device(self, devid):
        self.lib.hipSetDevice(devid)

    def mem_alloc(self, nbytes):
        return HIPDevAlloc(self, nbytes)

    def pagelocked_empty(self, shape, dtype):
        nbytes = np.prod(shape)*np.dtype(dtype).itemsize

        alloc = HIPHostAlloc(self, nbytes)
        alloc.__array_interface__ = {
            'version': 3,
            'typestr': np.dtype(dtype).str,
            'data': (int(alloc), False),
            'shape': tuple(shape)
        }

        return np.array(alloc, copy=False)

    def memcpy(self, dst, src, nbytes, stream=None):
        kind = self.lib.MEMCPY_DEFAULT

        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        if stream is None:
            self.lib.hipMemcpy(dst, src, nbytes, kind)
        else:
            self.lib.hipMemcpyAsync(dst, src, nbytes, kind, stream)

    def memset(self, dst, val, nbytes):
        self.lib.hipMemset(dst, val, nbytes)

    def load_module(self, code):
        return HIPModule(self, code)

    def create_stream(self):
        return HIPStream(self)

    def create_event(self):
        return HIPEvent(self)

    def create_graph(self):
        return HIPGraph(self)
