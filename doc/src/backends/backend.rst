*********
[backend]
*********

Parameterises the backend with

#. ``precision`` --- number precision, note not all backends support
   double precision:

    ``single`` | ``double``

#. ``memory-model`` --- if to enable support for large working sets;
    should be ``normal`` unless a memory-model error is encountered:

    ``normal`` | ``large``

#. ``collect-wait-times`` --- if to track MPI request wait times or not:

    ``True`` | ``False``

#. ``collect-wait-times-len`` --- size of the wait time history buffer:

     *int*

Example::

    [backend]
    precision = double
