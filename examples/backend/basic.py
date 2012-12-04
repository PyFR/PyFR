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

# Prepare the `ipadd` kernel which computes m1[i,j] = m1[i,j] + 2.0*m2[i,j]
addkern = be.kernel('ipadd', m1, m2)

# Get a kernel queue from the backend
q = be.queue()

# Execute the `addkern` kernel twice (or q % [addkern, addkern])
q % [addkern(2.0)]
q % [addkern(4.0)]

# Get m1 back as a numpy array
m1_n = m1.get()

# Take the difference between m1_n and (np1 + 4*np2)
res = np.sum(m1_n - (np1 + 2*np2 + 4*np2))

print 'Residual: {:.2g}'.format(res)
