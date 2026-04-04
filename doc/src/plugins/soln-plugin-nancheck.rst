**********************
[soln-plugin-nancheck]
**********************

Periodically checks the solution for NaN values. Parameterised with

#. ``nsteps`` --- check every ``nsteps``:

    *int*

#. ``trigger-set`` --- fire a named trigger when NaN is detected
   (optional); this can be used to gate a writer that dumps the
   solution before the abort:

    *string*

Example:

.. code-block:: ini

    [soln-plugin-nancheck]
    nsteps = 10
