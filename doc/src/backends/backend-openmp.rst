****************
[backend-openmp]
****************

Parameterises the OpenMP backend with

#. ``cc`` --- C compiler:

    *string*

#. ``cflags`` --- additional C compiler flags:

    *string*

#. ``alignb`` --- alignment requirement in bytes; must be a power of
   two and at least 32:

    *int*

#. ``schedule`` --- OpenMP loop scheduling scheme:

    ``static`` | ``dynamic`` | ``dynamic, n`` | ``guided`` | ``guided, n``

    where *n* is a positive integer.

Example::

    [backend-openmp]
    cc = gcc
