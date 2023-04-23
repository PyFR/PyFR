# Removing functionalities

Based on the last batch of iterations ...

## Excessive importance given to`intg._stabilise_final_n`

`ref-window` set in config file can conviniently be 8. There is no issue seen in the final optimisation. After all the optimisation is complete, a resulting candidate may fail after maybe half a flow pass. 

Solution to this issue is to increase the ref-window after the initial three phases of optimisation.

`intg._stabilise_final_n` does not seem to be used. 


## Consequetive model failure

Model has never failed consequetively, remove the part of the code that checks for this.

## Data deletion

Data is not removed right now. So remove all the code that deletes data.

## Increasing window

If increasing window, increase only after all optimisation complete. 
This is because one of the Re 500 seeds had failed after 1 flow pass.

## Online optimisation

If candidate is bad, only online version does something about it.

## Remove stability control

We are no longer deciding candidate quality based on its variance. So remove stability control.

## Remove `max-window` for capturing from congif file

