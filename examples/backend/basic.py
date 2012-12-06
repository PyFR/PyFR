#!/usr/bin/env python

"""Simple backend example

Demonstrates key backend concepts
"""

from pyfr.backends.cuda import CudaBackend
import numpy as np

# Number of rows and columns in our matrices
NROW, NCOL = 20, 70

# Generate two random NROW by NCOL matrices
np1 = np.random.randn(NROW, NCOL)
np2 = np.random.randn(NROW, NCOL)

# Create the backend
be = CudaBackend()

# Allocate matrices on the backend
m1 = be.matrix((NROW, NCOL), np1)
m2 = be.matrix((NROW, NCOL), np2)

# Prepare the `add` kernel
addkern = be.kernel('axnpby', m1, m2, m2)

# Get a kernel queue from the backend
q = be.queue()

# Execute the `addkern` kernel
q % [addkern(1.0, 2.0, 4.0)]

# Get m1 back as a numpy array
m1_n = m1.get()

# Take the difference
res = np.sum(m1_n - (np1 + 6*np2))

print 'Residual: {:.2g}'.format(res)
