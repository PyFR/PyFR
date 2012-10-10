======
Design
======

Introduction
------------

PyFR targets both conventional central processing units (CPUs) and
massively parallel graphics processing units (GPUs).  These hardware
platforms present very different programming interfaces.  Algorithms
and data structures which are optimal for one platform are usually not
so for the other.  As a consequence there is seldom an efficient 1:1
mapping between operations on a CPU to those on a GPU (and vice
versa).
