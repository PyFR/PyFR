# 2D Couette flow

To run the simulation with the OpenMP backend, you must use the appropriate
settings for your system. This is accomplished by editing the
`[backend-openmp]` section in the `couette_flow_2d.ini` configuration file.

For macOS:

```ini
[backend-openmp]
cc = gcc-8
cblas = /usr/lib/libblas.dylib
cblas-type = parallel
```

For Ubuntu:

```ini
[backend-openmp]
cc = gcc
cblas = /usr/lib/x86_64-linux-gnu/blas/libblas.so.3
cblas-type = parallel
```

## Run with OpenMP

Proceed with the following steps to run a serial 2D Couette flow simulation on
a mixed unstructured mesh using the OpenMP backend. These steps assume your
current working directory is this example's folder.

#### Step 1

Run pyfr to convert the Gmsh mesh file into a PyFR mesh file called `couette_flow_2d.pyfrm`.

```bash
$ pyfr import couette_flow_2d.msh couette_flow_2d.pyfrm
```

#### Step 2

Run pyfr to solve the Navier-Stokes equations on the mesh, generating a series
of PyFR solution files called `couette_flow_2d-*.pyfrs`.

```bash
$ pyfr run -b openmp -p couette_flow_2d.pyfrm couette_flow_2d.ini
```

#### Step 3

Run pyfr on the solution file `couette_flow_2d-040.pyfrs` converting it into
an unstructured VTK file called `couette_flow_2d-040.vtu`. Note that in order
to visualise the high-order data, each high-order element is sub-divided into
smaller linear elements. The level of sub-division is controlled by the
integer at the end of the command.

```bash
$ pyfr export couette_flow_2d.pyfrm couette_flow_2d-040.pyfrs couette_flow_2d-040.vtu -d 4
```

#### Step 4

Visualize the unstructured VTK file in Paraview.
