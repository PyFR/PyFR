<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

## Body-fixed walls are stationary in the rotating frame.
## Just use the standard slip adiabatic wall BC.
<%include file='pyfr.solvers.euler.kernels.bcs.slp-adia-wall'/>
