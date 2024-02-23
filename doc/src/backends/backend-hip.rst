*************
[backend-hip]
*************

Parameterises the HIP backend with

1. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``local-rank`` | ``uuid``

2. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``hip-aware``

3. ``rocblas-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 2048:

     *int*

4. ``gimmik-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 8:

    *int*

5. ``gimmik-nbench`` --- number of benchmarking runs for each
   kernel, defaults to 5:

     *int*

Example::

    [backend-hip]
    device-id = local-rank
    mpi-type = standard
