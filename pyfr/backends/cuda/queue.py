# -*- coding: utf-8 -*-

import collections

import pycuda.driver as cuda

from mpi4py import MPI

from pyfr.backends.base import iscomputekernel, ismpikernel, Queue


class CudaQueue(Queue):
    def __init__(self):
        # Last kernel we executed
        self._last = None

        # CUDA stream and MPI request list
        self._stream_comp = cuda.Stream()
        self._stream_copy = cuda.Stream()
        self._mpireqs = []

        # Items waiting to be executed
        self._items = collections.deque()

    def __lshift__(self, items):
        self._items.extend(items)

    def __mod__(self, items):
        self.run()
        self << items
        self.run()

    def _empty(self):
        return not self._items

    def _exec_item(self, item, rtargs):
        if iscomputekernel(item):
            item.run(self._stream_comp, self._stream_copy, *rtargs)
        elif ismpikernel(item):
            item.run(self._mpireqs, *rtargs)
        else:
            raise ValueError('Non compute/MPI kernel in queue')
        self._last = item

    def _exec_next(self):
        item, rtargs = self._items.popleft()

        # If we are at a sequence point then wait for current items
        if self._at_sequence_point(item):
            self._wait()

        # Execute the item
        self._exec_item(item, rtargs)

    def _exec_nowait(self):
        while self._items and not self._at_sequence_point(self._items[0][0]):
            self._exec_item(*self._items.popleft())

    def _wait(self):
        if iscomputekernel(self._last):
            self._stream_comp.synchronize()
            self._stream_copy.synchronize()
        elif ismpikernel(self._last):
            MPI.Prequest.Waitall(self._mpireqs)
            self._mpireqs = []
        self._last = None

    def _at_sequence_point(self, item):
        if (iscomputekernel(self._last) and not iscomputekernel(item)) or\
           (ismpikernel(self._last) and not ismpikernel(item)):
            return True
        else:
            return False

    def run(self):
        while self._items:
            self._exec_next()
        self._wait()

    @staticmethod
    def runall(queues):
        # First run any items which will not result in an implicit wait
        for q in queues:
            q._exec_nowait()

        # So long as there are items remaining in the queues
        while any(not q._empty() for q in queues):
            # Execute a (potentially) blocking item from each queue
            for q in [q for q in queues if not q._empty()]:
                q._exec_next()
                q._exec_nowait()

        # Wait for all tasks to complete
        for q in queues:
            q._wait()
