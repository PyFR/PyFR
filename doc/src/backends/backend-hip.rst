*************
[backend-hip]
*************

Parameterises the HIP backend with

#. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``local-rank`` | ``uuid``

#. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``hip-aware``

#. ``rocblas-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 2048:

     *int*

#. ``gimmik-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 8:

    *int*

#. ``gimmik-nbench`` --- number of benchmarking runs for each
   kernel, defaults to 5:

     *int*

#. ``lu-inverse`` --- kernels which invert the blocks of the
   ``block-jacobi`` preconditioner, defaults to ``mako``:

     ``mako`` | ``hip``

   where ``mako`` is the portable kernel PyFR generates for every
   backend and ``hip`` is a hand written kernel for AMD GPUs with
   MFMA (gfx90a, gfx94x and gfx95x), tuned on the MI300 series.  The
   environment variable ``PYFR_HIP_LU_INVERSE`` takes precedence over
   this option.

Example:

.. code-block:: ini

    [backend-hip]
    device-id = local-rank
    mpi-type = standard
