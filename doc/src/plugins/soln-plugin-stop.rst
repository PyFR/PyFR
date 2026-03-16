**********************
[soln-plugin-stop]
**********************

The stop plugin allows for the graceful termination of a simulation before the 
final time is reached. When triggered, the plugin issues a plugin_abort to the
integrator, deletes the trigger file, and exits the simulation cleanly. 
Parameterised with

1. ``file`` --- change the file from STOP to user-specified:

    *str*

Example::

    [soln-plugin-stop]
    file = ABORT
