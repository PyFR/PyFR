# -*- coding: utf-8 -*-

from ctypes import (POINTER, Structure, cast, c_char, c_char_p, c_int,
                    c_size_t, c_uint, c_void_p, pointer)

import numpy as np

from pyfr.ctypesutil import load_library


# Possible HIP exception types
HIPError = type('HIPError', (Exception, ), {})
HIPInvalidValue = type('HIPInvalidValue', (HIPError,), {})
HIPOutOfMemory = type('HIPOutOfMemory', (HIPError,), {})
HIPFileNotFound = type('HIPFileNotFound', (HIPError,), {})
HIPNotFound = type('HIPNotFound', (HIPError,), {})


class HIPDevProps(Structure):
    _fields_ = [
        ('s_name', c_char*256),
        ('i_total_global_mem', c_size_t),
        ('i_shared_mem_per_block', c_size_t),
        ('i_regs_per_block', c_int),
        ('i_warp_size', c_int),
        ('i_max_threads_per_block', c_int),
        ('ia_max_threads_dim', c_int*3),
        ('ia_max_grid_size', c_int*3),
        ('i_clock_rate', c_int),
        ('i_memory_clock_rate', c_int),
        ('i_memory_bus_width', c_int),
        ('i_total_const_mem', c_size_t),
        ('i_major', c_int),
        ('i_minor', c_int),
        ('i_multi_processor_count', c_int),
        ('i_l2_cache_size', c_int),
        ('i_max_threads_per_multiprocessor', c_int),
        ('i_compute_mode', c_int),
        ('i_clock_instruction_rate', c_int),
        ('b_global_int32_atomics', c_uint, 1),
        ('b_global_float_atomic_exch', c_uint, 1),
        ('b_shared_int32_atomics', c_uint, 1),
        ('b_shared_float_atomic_exch', c_uint, 1),
        ('b_float_atomic_add', c_uint, 1),
        ('b_global_int64_atomics', c_uint, 1),
        ('b_shared_int64_atomics', c_uint, 1),
        ('b_doubles', c_uint, 1),
        ('b_warp_vote', c_uint, 1),
        ('b_wrap_ballot', c_uint, 1),
        ('b_warp_shuffle', c_uint, 1),
        ('b_funnel_shift', c_uint, 1),
        ('b_thread_fence_system', c_uint, 1),
        ('b_sync_threads_ext', c_uint, 1),
        ('b_surface_funcs', c_uint, 1),
        ('b_3d_grid', c_uint, 1),
        ('b_dynamic_parallelism', c_uint, 1),
        ('b_concurrent_kernels', c_int),
        ('i_pci_domain_id', c_int),
        ('i_pci_bus_id', c_int),
        ('i_pci_device_id', c_int),
        ('i_max_shared_memory_per_multiprocessor', c_size_t),
        ('b_is_multi_gpu_board', c_int),
        ('b_can_map_host_memory', c_int),
        ('i_gcn_arch', c_int),
        ('s_gcn_arch_name', c_char*256),
        ('b_integrated', c_int),
        ('b_cooperative_launch', c_int),
        ('b_cooperative_multi_device_launch', c_int),
        ('i_max_texture_1d', c_int),
        ('ia_max_texture_2d', c_int*2),
        ('ia_max_texture_3d', c_int*3),
        ('hdp_mem_flush_cntl', c_void_p),
        ('hdp_reg_flush_cntl', c_void_p),
        ('i_mem_pitch', c_size_t),
        ('i_texture_alignment', c_size_t),
        ('i_texture_pitch_alignment', c_size_t),
        ('b_kernel_exec_timeout_enalbed', c_int),
        ('b_ecc_enalbed', c_int),
        ('b_tcc_driver', c_int),
        ('b_cooperative_multi_device_unmatched_func', c_int),
        ('b_cooperative_multi_device_unmatched_grid_dim', c_int),
        ('b_cooperative_multi_device_unmatched_block_dim', c_int),
        ('b_cooperative_multi_device_unmatched_shared_mem', c_int),
        ('b_is_large_bar', c_int),
        ('i_asic_revision', c_int),
        ('b_managed_memory', c_int),
        ('b_direct_managed_mem_access_from_host', c_int),
        ('b_concurrent_managed_access', c_int),
        ('b_pageable_memory_access', c_int),
        ('b_pageable_memory_access_using_page_tables', c_int),
        ('reserved', c_int*64)
    ]


class HIPWrappers(object):
    # Possible return codes
    _statuses = {
        1: HIPInvalidValue,
        2: HIPOutOfMemory,
        301: HIPFileNotFound,
        500: HIPNotFound
    }

    # Constants
    hipMemcpyDefault = 4

    def __init__(self):
        lib = load_library('amdhip64')

        # hipGetDeviceProperties
        self.hipGetDeviceProperties = lib.hipGetDeviceProperties
        self.hipGetDeviceProperties.argtypes = [POINTER(HIPDevProps), c_int]
        self.hipGetDeviceProperties.errcheck = self._errcheck

        # hipSetDevice
        self.hipSetDevice = lib.hipSetDevice
        self.hipSetDevice.argtypes = [c_int]
        self.hipSetDevice.errcheck = self._errcheck

        # hipMalloc
        self.hipMalloc = lib.hipMalloc
        self.hipMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        self.hipMalloc.errcheck = self._errcheck

        # hipFree
        self.hipFree = lib.hipFree
        self.hipFree.argtypes = [c_void_p]
        self.hipFree.errcheck = self._errcheck

        # hipHostMalloc
        self.hipHostMalloc = lib.hipHostMalloc
        self.hipHostMalloc.argtypes = [POINTER(c_void_p), c_size_t, c_uint]
        self.hipHostMalloc.errcheck = self._errcheck

        # hipHostFree
        self.hipHostFree = lib.hipHostFree
        self.hipHostFree.argtypes = [c_void_p]
        self.hipHostFree.errcheck = self._errcheck

        # hipMemcpy
        self.hipMemcpy = lib.hipMemcpy
        self.hipMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
        self.hipMalloc.errcheck = self._errcheck

        # hipMemcpyAsync
        self.hipMemcpyAsync = lib.hipMemcpyAsync
        self.hipMemcpyAsync.argtypes = [c_void_p, c_void_p, c_size_t,
                                        c_int, c_void_p]
        self.hipMemcpyAsync.errcheck = self._errcheck

        # hipMemset
        self.hipMemset = lib.hipMemset
        self.hipMemset.argtypes = [c_void_p, c_int, c_size_t]
        self.hipMemset.errcheck = self._errcheck

        # hipStreamCreate
        self.hipStreamCreate = lib.hipStreamCreate
        self.hipStreamCreate.argtypes = [POINTER(c_void_p)]
        self.hipStreamCreate.errcheck = self._errcheck

        # hipStreamDestroy
        self.hipStreamDestroy = lib.hipStreamDestroy
        self.hipStreamDestroy.argtypes = [c_void_p]
        self.hipStreamDestroy.errcheck = self._errcheck

        # hipStreamSynchronize
        self.hipStreamSynchronize = lib.hipStreamSynchronize
        self.hipStreamSynchronize.argtypes = [c_void_p]
        self.hipStreamSynchronize.errcheck = self._errcheck

        # hipStreamWaitEvent
        self.hipStreamWaitEvent = lib.hipStreamWaitEvent
        self.hipStreamWaitEvent.argtypes = [c_void_p, c_void_p, c_uint]
        self.hipStreamWaitEvent.errcheck = self._errcheck

        # hipEventCreate
        self.hipEventCreate = lib.hipEventCreate
        self.hipEventCreate.argtypes = [POINTER(c_void_p)]
        self.hipEventCreate.errcheck = self._errcheck

        # hipEventDestroy
        self.hipEventDestroy = lib.hipEventDestroy
        self.hipEventDestroy.argtypes = [c_void_p]
        self.hipEventDestroy.errcheck = self._errcheck

        # hipEventRecord
        self.hipEventRecord = lib.hipEventRecord
        self.hipEventRecord.argtypes = [c_void_p, c_void_p]
        self.hipEventRecord.errcheck = self._errcheck

        # hipModuleLoad
        self.hipModuleLoad = lib.hipModuleLoad
        self.hipModuleLoad.argtypes = [POINTER(c_void_p), c_char_p]
        self.hipModuleLoad.errcheck = self._errcheck

        # hipModuleUnload
        self.hipModuleUnload = lib.hipModuleUnload
        self.hipModuleUnload.argtypes = [c_void_p]
        self.hipModuleUnload.errcheck = self._errcheck

        # hipModuleGetFunction
        self.hipModuleGetFunction = lib.hipModuleGetFunction
        self.hipModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p,
                                              c_char_p]
        self.hipModuleGetFunction.errcheck = self._errcheck

        # hipModuleLaunchKernel
        self.hipModuleLaunchKernel = lib.hipModuleLaunchKernel
        self.hipModuleLaunchKernel.argtypes = [
            c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
            c_void_p, POINTER(c_void_p), c_void_p
        ]


    def _errcheck(self, status, fn, args):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise HIPError


class _HIPBase(object):
    _destroyfn = None

    def __init__(self, lib, ptr):
        self.lib = lib
        self._as_parameter_ = ptr.value

    def __del__(self):
        if self._destroyfn:
            try:
                getattr(self.lib, self._destroyfn)(self)
            except AttributeError:
                pass

    def __int__(self):
        return self._as_parameter_


class HIPDevAlloc(_HIPBase):
    _destroyfn = 'hipFree'

    def __init__(self, lib, nbytes):
        ptr = c_void_p()
        lib.hipMalloc(ptr, nbytes)

        super().__init__(lib, ptr)
        self.nbytes = nbytes


class HIPHostAlloc(_HIPBase):
    _destroyfn = 'hipHostFree'

    def __init__(self, lib, nbytes):
        self.nbytes = nbytes

        ptr = c_void_p()
        lib.hipHostMalloc(ptr, nbytes, 0)

        super().__init__(lib, ptr)


class HIPStream(_HIPBase):
    _destroyfn = 'hipStreamDestroy'

    def __init__(self, lib):
        ptr = c_void_p()
        lib.hipStreamCreate(ptr)

        super().__init__(lib, ptr)

    def synchronize(self):
        self.lib.hipStreamSynchronize(self)

    def wait_for_event(self, event):
        self.lib.hipStreamWaitEvent(self, event, 0)


class HIPEvent(_HIPBase):
    _destroyfn = 'hipEventDestroy'

    def __init__(self, lib):
        ptr = c_void_p()
        lib.hipEventCreate(ptr)

        super().__init__(lib, ptr)

    def record(self, stream):
        self.lib.hipEventRecord(self, stream)


class HIPModule(_HIPBase):
    _destroyfn = 'hipModuleUnload'

    def __init__(self, lib, path):
        ptr = c_void_p()
        lib.hipModuleLoad(ptr, path.encode())

        super().__init__(lib, ptr)

    def get_function(self, name, argspec):
        return HIPFunction(self.lib, self, name, argspec)


class HIPFunction(_HIPBase):
    def __init__(self, lib, module, name, argtypes):
        ptr = c_void_p()
        lib.hipModuleGetFunction(ptr, module, name.encode())

        super().__init__(lib, ptr)

        # Save a reference to our underlying module
        self.module = module

        # For each argument type instantiate the corresponding ctypes type
        self._args = args = [atype() for atype in argtypes]

        # Obtain pointers to these arguments
        self._arg_ptrs = [cast(pointer(arg), c_void_p) for arg in args]
        self._arg_ptrs = (c_void_p * len(args))(*self._arg_ptrs)

    def exec_async(self, grid, block, stream, *args):
        for src, dst in zip(args, self._args):
            dst.value = getattr(src, '_as_parameter_', src)

        self.lib.hipModuleLaunchKernel(self, *grid, *block, 0, stream,
                                       self._arg_ptrs, 0)


class HIP(object):
    def __init__(self):
        self._wrappers = HIPWrappers()

    def set_device(self, devid):
        self._wrappers.hipSetDevice(devid)

    def device_properties(self, devid):
        props = HIPDevProps()
        self._wrappers.hipGetDeviceProperties(props, devid)

        dprops = {}
        for name, *t in props._fields_:
            if name.startswith('b_'):
                dprops[name[2:]] = bool(getattr(props, name))
            elif name.startswith('i_'):
                dprops[name[2:]] = int(getattr(props, name))
            elif name.startswith('ia_'):
                dprops[name[3:]] = [int(v) for v in getattr(props, name)]
            elif name.startswith('s_'):
                dprops[name[2:]] = getattr(props, name).decode()

        return dprops

    def mem_alloc(self, nbytes):
        return HIPDevAlloc(self._wrappers, nbytes)

    def pagelocked_empty(self, shape, dtype):
        nbytes = np.prod(shape)*np.dtype(dtype).itemsize

        alloc = HIPHostAlloc(self._wrappers, nbytes)
        alloc.__array_interface__ = {
            'version': 3,
            'typestr': np.dtype(dtype).str,
            'data': (int(alloc), False),
            'shape': tuple(shape)
        }

        return np.array(alloc, copy=False)

    def memcpy(self, dst, src, nbytes):
        flags = self._wrappers.hipMemcpyDefault

        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        self._wrappers.hipMemcpy(dst, src, nbytes, flags)

    def memcpy_async(self, dst, src, nbytes, stream):
        flags = self._wrappers.hipMemcpyDefault

        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        self._wrappers.hipMemcpyAsync(dst, src, nbytes, flags, stream)

    def memset(self, dst, val, nbytes):
        self._wrappers.hipMemset(dst, val, nbytes)

    def load_module(self, path):
        return HIPModule(self._wrappers, path)

    def create_stream(self):
        return HIPStream(self._wrappers)

    def create_event(self):
        return HIPEvent(self._wrappers)
