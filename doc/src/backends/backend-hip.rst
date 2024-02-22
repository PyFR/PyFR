*************
[backend-hip]
*************

Parameterises the HIP backend with

1. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``local-rank`` | ``uuid``

2. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``hip-aware``

3. ``rocblas-nkerns`` --- maximum number of kernel algorithms to try, defaults
   to 2048:

     *int*

Example::

    [backend-hip]
    device-id = local-rank
    mpi-type = standard
