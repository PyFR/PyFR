<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

## Body-fixed walls are stationary in the rotating frame.
## Just use the standard no-slip adiabatic wall BC.
<%include file='pyfr.solvers.navstokes.kernels.bcs.no-slp-adia-wall'/>
