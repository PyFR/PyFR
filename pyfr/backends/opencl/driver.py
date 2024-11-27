from ctypes import (POINTER, byref, cast, create_string_buffer, c_char,
                    c_char_p, c_double, c_float, c_int, c_int32, c_int64,
                    c_size_t, c_uint, c_uint64, c_ulong, c_void_p, sizeof)
from uuid import UUID

import numpy as np

from pyfr.ctypesutil import LibWrapper


# Possible OpenCL exception types
class OpenCLError(Exception): pass
class OpenCLDeviceNotFound(OpenCLError): pass
class OpenCLDeviceNotAvailable(OpenCLError): pass
class OpenCLAllocationFailure(OpenCLError): pass
class OpenCLOutOfResources(OpenCLError): pass
class OpenCLBuildProgramFailure(OpenCLError): pass
class OpenCLMisalignedSubBufferOffset(OpenCLError): pass
class OpenCLDevicePartitioningFailed(OpenCLError): pass
class OpenCLInvalidValue(OpenCLError): pass
class OpenCLInvalidKernelName(OpenCLError): pass
class OpenCLInvalidKernelArgs(OpenCLError): pass
class OpenCLInvalidWorkGroupSize(OpenCLError): pass
class OpenCLInvalidWorkItemSize(OpenCLError): pass
class OpenCLInvalidGlobalWorkSize(OpenCLError): pass


class OpenCLWrappers(LibWrapper):
    _libname = 'OpenCL'

    # Error codes
    _statuses = {
        -1: OpenCLDeviceNotFound,
        -2: OpenCLDeviceNotAvailable,
        -4: OpenCLAllocationFailure,
        -5: OpenCLOutOfResources,
        -11: OpenCLBuildProgramFailure,
        -13: OpenCLMisalignedSubBufferOffset,
        -18: OpenCLDevicePartitioningFailed,
        -30: OpenCLInvalidValue,
        -46: OpenCLInvalidKernelName,
        -52: OpenCLInvalidKernelArgs,
        -54: OpenCLInvalidWorkGroupSize,
        -55: OpenCLInvalidWorkItemSize,
        -63: OpenCLInvalidGlobalWorkSize,
        '*': OpenCLError
    }

    # Constants
    BUFFER_CREATE_TYPE_REGION = 0x1220
    DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = 0x20
    DEVICE_EXTENSIONS = 0x1030
    DEVICE_LOCAL_MEM_SIZE = 0x1023
    DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019
    DEVICE_NAME = 0x102b
    DEVICE_VENDOR = 0x102c
    DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088
    DEVICE_TYPE_ACCELERATOR = 0x8
    DEVICE_TYPE_ALL = 0xffffffff
    DEVICE_TYPE_CPU = 0x2
    DEVICE_TYPE_GPU = 0x4
    DEVICE_UUID = 0x106a
    DRIVER_VERSION = 0x102d
    MAP_READ = 0x1
    MAP_WRITE = 0x2
    MEM_READ_WRITE = 0x1
    MEM_ALLOC_HOST_PTR = 0x10
    PLATFORM_NAME = 0x0902
    PROGRAM_BINARY_SIZES = 0x1165
    PROGRAM_BINARIES = 0x1166
    PROGRAM_BUILD_LOG = 0x1183
    PROFILING_COMMAND_START = 0x1282
    PROFILING_COMMAND_END = 0x1283
    QUEUE_OUT_OF_ORDER = 0x1
    QUEUE_PROFILING = 0x2
    QUEUE_PROPERTIES = 0x1093

    # Functions
    _functions = [
        (c_int, 'clGetPlatformIDs', c_uint, POINTER(c_void_p),
         POINTER(c_uint)),
        (c_int, 'clGetPlatformInfo', c_void_p, c_uint, c_size_t, c_void_p,
         POINTER(c_size_t)),
        (c_int, 'clGetDeviceIDs', c_void_p, c_uint64, c_uint,
         POINTER(c_void_p), POINTER(c_uint)),
        (c_int, 'clGetDeviceInfo', c_void_p, c_uint, c_size_t, c_void_p,
         POINTER(c_size_t)),
        (c_int, 'clCreateSubDevices', c_void_p, POINTER(c_uint64), c_uint,
         POINTER(c_void_p), POINTER(c_uint)),
        (c_int, 'clReleaseDevice', c_void_p),
        (c_void_p, 'clCreateContext', c_void_p, c_uint, POINTER(c_void_p),
         c_void_p, c_void_p, POINTER(c_int)),
        (c_int, 'clReleaseContext', c_void_p),
        (c_void_p, 'clCreateCommandQueueWithProperties', c_void_p, c_void_p,
         POINTER(c_uint64), POINTER(c_int)),
        (c_int, 'clReleaseCommandQueue', c_void_p),
        (c_int, 'clFinish', c_void_p),
        (c_int, 'clFlush', c_void_p),
        (c_int, 'clGetEventProfilingInfo', c_void_p, c_uint, c_size_t,
         c_void_p, POINTER(c_size_t)),
        (c_int, 'clReleaseEvent', c_void_p),
        (c_int, 'clWaitForEvents', c_uint, c_void_p),
        (c_void_p, 'clCreateBuffer', c_void_p, c_uint64, c_size_t, c_void_p,
         POINTER(c_int)),
        (c_void_p, 'clCreateSubBuffer', c_void_p, c_uint64, c_uint, c_void_p,
         POINTER(c_int)),
        (c_int, 'clReleaseMemObject', c_void_p),
        (c_int, 'clEnqueueReadBuffer', c_void_p, c_void_p, c_uint, c_size_t,
         c_size_t, c_void_p, c_uint, POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'clEnqueueWriteBuffer', c_void_p, c_void_p, c_uint, c_size_t,
         c_size_t, c_void_p, c_uint, POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'clEnqueueCopyBuffer', c_void_p, c_void_p, c_void_p, c_size_t,
         c_size_t, c_size_t, c_uint, POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'clEnqueueFillBuffer', c_void_p, c_void_p, c_void_p, c_size_t,
         c_size_t, c_size_t, c_uint, POINTER(c_void_p), POINTER(c_void_p)),
        (c_void_p, 'clEnqueueMapBuffer', c_void_p, c_void_p, c_uint, c_uint64,
         c_size_t, c_size_t, c_uint, POINTER(c_void_p), POINTER(c_void_p),
         POINTER(c_int)),
        (c_int, 'clEnqueueUnmapMemObject', c_void_p, c_void_p, c_void_p,
         c_uint, POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'clEnqueueMarkerWithWaitList', c_void_p, c_uint,
         POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'clEnqueueBarrierWithWaitList', c_void_p, c_uint,
         POINTER(c_void_p), POINTER(c_void_p)),
        (c_void_p, 'clCreateProgramWithSource', c_void_p, c_uint,
         POINTER(c_char_p), POINTER(c_size_t), POINTER(c_int)),
        (c_void_p, 'clCreateProgramWithBinary', c_void_p, c_uint,
         POINTER(c_void_p), POINTER(c_size_t), POINTER(c_char_p),
         POINTER(c_int), POINTER(c_int)),
        (c_int, 'clReleaseProgram', c_void_p),
        (c_int, 'clBuildProgram', c_void_p, c_uint, POINTER(c_void_p),
         c_char_p, c_void_p, c_void_p),
        (c_int, 'clGetProgramInfo', c_void_p, c_int, c_size_t, c_void_p,
         POINTER(c_size_t)),
        (c_int, 'clGetProgramBuildInfo', c_void_p, c_void_p, c_uint, c_size_t,
         c_void_p, POINTER(c_size_t)),
        (c_void_p, 'clCreateKernel', c_void_p, c_char_p, POINTER(c_int)),
        (c_void_p, 'clCloneKernel', c_void_p, c_int),
        (c_int, 'clReleaseKernel', c_void_p),
        (c_int, 'clEnqueueNDRangeKernel', c_void_p, c_void_p, c_uint,
         POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t), c_uint,
         POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'clSetKernelArg', c_void_p, c_uint, c_size_t, c_void_p)
    ]

    def __init__(self):
        super().__init__()

        # Automate error checking for functions which signal their status via
        # output arguments rather than return values
        for fret, fname, *fargs in self._functions:
            if fret == c_void_p:
                setattr(self, fname, self._argerrcheck(getattr(self, fname)))

    def _argerrcheck(self, fn):
        def newfn(*args):
            err = c_int()
            ret = fn(*args, err)

            if err.value != 0:
                try:
                    raise self._statuses[err.value]
                except KeyError:
                    raise self._statuses['*'] from None

            return ret

        return newfn


class _OpenCLBase:
    _destroyfn = None

    def __init__(self, lib, ptr):
        self.lib = lib
        self._as_parameter_ = getattr(ptr, 'value', ptr)

    def __del__(self):
        if self._destroyfn:
            try:
                getattr(self.lib, self._destroyfn)(self)
            except AttributeError:
                pass

    def __int__(self):
        return self._as_parameter_


class _OpenCLWaitFor:
    def _make_wait_for(self, events):
        if events:
            nwait = len(events)
            events = [int(e) for e in events]

            try:
                wait_for = self._wait_for_ptrs
                wait_for[:nwait] = events
            except (AttributeError, ValueError):
                self._wait_for_ptrs = wait_for = (c_void_p * nwait)()
                wait_for[:] = events

            return nwait, wait_for
        else:
            return 0, None


class OpenCLPlatform(_OpenCLBase):
    def __init__(self, lib, ptr):
        super().__init__(lib, ptr)

        self.name = self._query_str('name')

    def get_devices(self, devtype='all'):
        devtype = getattr(self.lib, f'DEVICE_TYPE_{devtype.upper()}')
        ndevices = c_uint()

        try:
            self.lib.clGetDeviceIDs(self, devtype, 0, None, ndevices)
        except OpenCLDeviceNotFound:
            return []

        devices = (c_void_p * ndevices.value)()
        self.lib.clGetDeviceIDs(self, devtype, ndevices, devices, None)

        return [OpenCLDevice(self.lib, d) for d in devices]

    def _query_str(self, param):
        param = getattr(self.lib, f'PLATFORM_{param.upper()}')

        nbytes = c_size_t()
        self.lib.clGetPlatformInfo(self, param, 0, None, nbytes)

        buf = create_string_buffer(nbytes.value)
        self.lib.clGetPlatformInfo(self, param, nbytes, buf, None)

        return buf.value.decode()


class OpenCLDevice(_OpenCLBase):
    _destroyfn = 'clReleaseDevice'

    def __init__(self, lib, ptr):
        super().__init__(lib, ptr)

        self.name = self._query_str('name')
        self.vendor = self._query_str('vendor')

        self.local_mem_size = self._query_type(c_ulong, 'local_mem_size')
        self.mem_align = self._query_type(c_uint, 'mem_base_addr_align') // 8

        self.driver_version = self._query_str('version', prefix='driver')

        self.extensions = set(self._query_str('extensions').split())
        self.has_fp64 = 'cl_khr_fp64' in self.extensions

        if 'cl_khr_device_uuid' in self.extensions:
            self.uuid = UUID(bytes=self._query_type(c_char*16, 'uuid'))
        else:
            self.uuid = None

    def subdevices(self):
        lib = self.lib
        params = (c_uint64 * 3)(lib.DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                                lib.DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
                                0)
        nsubdevices = c_uint()

        try:
            lib.clCreateSubDevices(self, params, 0, None, nsubdevices)
        except OpenCLDevicePartitioningFailed:
            return []

        subdevices = (c_void_p * nsubdevices.value)
        lib.clCreateSubDevices(self, params, nsubdevices, subdevices, None)

        return [OpenCLDevice(lib, d) for d in subdevices]

    def _query_type(self, type_t, param, prefix='device'):
        param = getattr(self.lib, f'{prefix.upper()}_{param.upper()}')

        v = type_t()
        self.lib.clGetDeviceInfo(self, param, sizeof(v), byref(v), None)

        return v.raw if hasattr(v, 'raw') else v.value

    def _query_str(self, param, prefix='device'):
        param = getattr(self.lib, f'{prefix.upper()}_{param.upper()}')

        nbytes = c_size_t()
        self.lib.clGetDeviceInfo(self, param, 0, None, nbytes)

        buf = create_string_buffer(nbytes.value)
        self.lib.clGetDeviceInfo(self, param, nbytes, buf, None)

        return buf.value.decode()


class OpenCLBuffer(_OpenCLBase):
    _destroyfn = 'clReleaseMemObject'

    def __init__(self, lib, ctx, nbytes):
        self.nbytes = nbytes

        ptr = lib.clCreateBuffer(ctx, lib.MEM_READ_WRITE, nbytes, None)

        super().__init__(lib, ptr)

    def slice(self, off, nbytes):
        return OpenCLSubBuffer(self.lib, self, off, nbytes)


class OpenCLSubBuffer(_OpenCLBase):
    _destroyfn = 'clReleaseMemObject'

    def __init__(self, lib, buf, off, nbytes):
        self.parent = buf
        self.nbytes = nbytes

        rgn = (c_size_t * 2)(off, nbytes)
        ptr = lib.clCreateSubBuffer(buf, lib.MEM_READ_WRITE,
                                    lib.BUFFER_CREATE_TYPE_REGION, rgn)

        super().__init__(lib, ptr)


class OpenCLHostAlloc(_OpenCLBase):
    _destroyfn = 'clReleaseMemObject'

    def __init__(self, lib, ctx, queue, nbytes):
        self.qdflt = queue
        self.nbytes = nbytes

        flags = lib.MEM_READ_WRITE | lib.MEM_ALLOC_HOST_PTR
        ptr = lib.clCreateBuffer(ctx, flags, nbytes, None)

        super().__init__(lib, ptr)

        flags = lib.MAP_READ | lib.MAP_WRITE
        self.host_ptr = lib.clEnqueueMapBuffer(queue, ptr, True, flags, 0,
                                               nbytes, 0, None, None)

    def __del__(self):
        if hasattr(self, 'host_ptr'):
            self.lib.clEnqueueUnmapMemObject(self.qdflt, self, self.host_ptr,
                                             0, None, None)

        super().__del__()


class OpenCLEvent(_OpenCLBase):
    _destroyfn = 'clReleaseEvent'

    def _profile_param(self, param):
        t = c_ulong()
        self.lib.clGetEventProfilingInfo(self, param, sizeof(t), byref(t),
                                         None)

        return t.value / 1e9

    @property
    def start_time(self):
        return self._profile_param(self.lib.PROFILING_COMMAND_START)

    @property
    def end_time(self):
        return self._profile_param(self.lib.PROFILING_COMMAND_END)


class OpenCLQueue(_OpenCLWaitFor, _OpenCLBase):
    _destroyfn = 'clReleaseCommandQueue'

    def __init__(self, lib, ctx, dev, out_of_order, profiling):
        self.out_of_order = out_of_order
        self.profiling = profiling

        if out_of_order or profiling:
            flags = lib.QUEUE_OUT_OF_ORDER if out_of_order else 0
            flags |= lib.QUEUE_PROFILING if profiling else 0

            props = (c_uint64 * 3)(lib.QUEUE_PROPERTIES, flags, 0)
        else:
            props = None

        ptr = lib.clCreateCommandQueueWithProperties(ctx, dev, props)

        super().__init__(lib, ptr)

    def marker(self, wait_for=None):
        evt_ptr = c_void_p()
        wait_for = self._make_wait_for(wait_for)

        self.lib.clEnqueueMarkerWithWaitList(self, *wait_for, evt_ptr)

        return OpenCLEvent(self.lib, evt_ptr)

    def barrier(self, wait_for=None):
        wait_for = self._make_wait_for(wait_for)

        self.lib.clEnqueueBarrierWithWaitList(self, *wait_for, None)

    def finish(self):
        self.lib.clFinish(self)

    def flush(self):
        self.lib.clFlush(self)


class OpenCLProgram(_OpenCLBase):
    _destroyfn = 'clReleaseProgram'

    def __init__(self, lib, ctx, dev, src, flags):
        devptr = c_void_p(dev._as_parameter_)

        match src:
            # Binary program
            case bytes():
                buf = c_char_p(src)
                nbytes = c_size_t(len(src))
                ptr = lib.clCreateProgramWithBinary(ctx, 1, devptr, nbytes,
                                                    byref(buf), None)

                flags = None
            # Source program
            case str():
                buf = c_char_p(src.encode())
                ptr = lib.clCreateProgramWithSource(ctx, 1, buf, None)

                flags = c_char_p(' '.join(flags).encode())

        super().__init__(lib, ptr)

        try:
            lib.clBuildProgram(self, 1, devptr, flags, None, None)
        except OpenCLBuildProgramFailure:
            nbytes = c_size_t()
            lib.clGetProgramBuildInfo(self, devptr, lib.PROGRAM_BUILD_LOG, 0,
                                      None, nbytes)

            buf = create_string_buffer(nbytes.value)
            lib.clGetProgramBuildInfo(self, devptr, lib.PROGRAM_BUILD_LOG,
                                      nbytes, buf, None)

            raise OpenCLBuildProgramFailure(buf.value.decode()) from None

    def get_kernel(self, name, argtypes):
        ptr = self.lib.clCreateKernel(self, name.encode())
        return OpenCLKernel(self.lib, ptr, argtypes)

    def get_binary(self):
        nbytes = c_size_t()
        self.lib.clGetProgramInfo(self, self.lib.PROGRAM_BINARY_SIZES,
                                  sizeof(nbytes), byref(nbytes), None)

        buf = create_string_buffer(nbytes.value)
        self.lib.clGetProgramInfo(self, self.lib.PROGRAM_BINARIES,
                                  sizeof(c_void_p), byref(cast(buf, c_void_p)),
                                  None)

        return buf.raw


class OpenCLKernel(_OpenCLWaitFor, _OpenCLBase):
    _destroyfn = 'clReleaseKernel'

    typemap = [c_double, c_float, c_int32, c_int64, c_uint64]
    typemap = {k: (k(), sizeof(k)) for k in typemap}

    def __init__(self, lib, ptr, argtypes):
        super().__init__(lib, ptr)

        self.argtypes = argtypes

        # For each argument type fetch the corresponding ctypes instance
        self._argsz = [self.typemap[atype] for atype in argtypes]

    def clone(self):
        ptr = self.lib.clCloneKernel(self)
        return OpenCLKernel(self.lib, ptr, self.argtypes)

    def set_arg(self, i, v):
        arg, sz = self._argsz[i]
        arg.value = getattr(v, '_as_parameter_', v)

        self.lib.clSetKernelArg(self, i, sz, byref(arg))

    def set_args(self, *kargs, start=0):
        for i, v in enumerate(kargs, start=start):
            self.set_arg(i, v)

    def set_dims(self, gs, ls=None):
        self._gs = (c_size_t * len(gs))(*gs)
        self._ls = (c_size_t * len(ls))(*ls) if ls else None
        self._wd = len(gs)

    def exec_async(self, queue, wait_for=None, ret_evt=False):
        evt_ptr = c_void_p() if ret_evt else None
        wait_for = self._make_wait_for(wait_for)

        self.lib.clEnqueueNDRangeKernel(queue, self, self._wd, None, self._gs,
                                        self._ls, *wait_for, evt_ptr)

        if ret_evt:
            return OpenCLEvent(self.lib, evt_ptr)


class OpenCL(_OpenCLWaitFor):
    def __init__(self):
        self.ctx = None
        self.lib = OpenCLWrappers()

    def __del__(self):
        if self.ctx:
            self.lib.clReleaseContext(self.ctx)

    def get_platforms(self):
        nplatforms = c_uint()
        self.lib.clGetPlatformIDs(0, None, nplatforms)

        platforms = (c_void_p * nplatforms.value)()
        self.lib.clGetPlatformIDs(nplatforms, platforms, None)

        return [OpenCLPlatform(self.lib, p) for p in platforms]

    def set_device(self, dev):
        if self.ctx:
            raise RuntimeError('Device has already been set')

        devptr = c_void_p(dev._as_parameter_)
        self.ctx = self.lib.clCreateContext(None, 1, devptr, None, None)
        self.dev = dev

        # Allocate a default queue
        self.qdflt = self.queue()

    def mem_alloc(self, nbytes):
        return OpenCLBuffer(self.lib, self.ctx, nbytes)

    def pagelocked_empty(self, shape, dtype):
        nbytes = np.prod(shape)*np.dtype(dtype).itemsize

        alloc = OpenCLHostAlloc(self.lib, self.ctx, self.qdflt, nbytes)
        alloc.__array_interface__ = {
            'version': 3,
            'typestr': np.dtype(dtype).str,
            'data': (alloc.host_ptr, False),
            'shape': tuple(shape)
        }

        return np.array(alloc, copy=False)

    def zero(self, dst, off, nbytes):
        z = c_char(0)
        self.lib.clEnqueueFillBuffer(self.qdflt, dst, byref(z), 1, off,
                                     nbytes, 0, None, None)
        self.qdflt.finish()

    def memcpy(self, queue, dst, src, nbytes, blocking=False, wait_for=None,
               ret_evt=False):
        evt_ptr = c_void_p() if ret_evt else None
        wait_for = self._make_wait_for(wait_for)

        # Device to host
        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

            self.lib.clEnqueueReadBuffer(queue, src, blocking, 0, nbytes,
                                         dst, *wait_for, evt_ptr)
        # Host to device
        elif isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

            self.lib.clEnqueueWriteBuffer(queue, dst, blocking, 0, nbytes,
                                          src, *wait_for, evt_ptr)
        # Device to device
        else:
            self.lib.clEnqueueCopyBuffer(queue, src, dst, 0, 0, nbytes,
                                         *wait_for, evt_ptr)

        if ret_evt:
            return OpenCLEvent(self.lib, evt_ptr)

    def program(self, src, flags=None):
        return OpenCLProgram(self.lib, self.ctx, self.dev, src, flags or [])

    def event(self, evt):
        return OpenCLEvent(self.lib, evt)

    def wait_for_events(self, events):
        self.lib.clWaitForEvents(*self._make_wait_for(events))

    def queue(self, out_of_order=False, profiling=False):
        return OpenCLQueue(self.lib, self.ctx, self.dev, out_of_order,
                           profiling)
