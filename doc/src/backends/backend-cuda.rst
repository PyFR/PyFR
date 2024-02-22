**************
[backend-cuda]
**************

Parameterises the CUDA backend with

1. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``round-robin`` | ``local-rank`` | ``uuid``

2. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``cuda-aware``

3. ``cflags`` --- additional NVIDIA realtime compiler (``nvrtc``) flags:

    *string*

Example::

    [backend-cuda]
    device-id = round-robin
    mpi-type = standard
