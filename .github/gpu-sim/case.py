"""Small PyFR case for the simulated-GPU CI job (.github/workflows/gpu-sim.yml).

A Gaussian pressure/density pulse in a closed unit square, 6x6 quads, run to
t = 0.1 with RK4. Sub-commands:

  case.py mesh OUT.msh            write a Gmsh 2.2 mesh
  case.py config SYSTEM ORDER OUT write a PyFR config (euler or navier-stokes)
  case.py check FILE.pyfrs KEY    check the solution against reference values
"""

import sys

import h5py
import numpy as np

NCELL = 6

CONFIG = """\
[backend]
precision = double

[backend-cuda]
cflags = --gpu-architecture=compute_90

[constants]
gamma = 1.4
mu = 0.01
Pr = 0.72

[solver]
system = {system}
order = {order}

[solver-time-integrator]
scheme = rk4
controller = none
tstart = 0.0
tend = 0.1
dt = 0.002

[solver-interfaces]
riemann-solver = rusanov
ldg-beta = 0.5
ldg-tau = 0.1

[solver-interfaces-line]
flux-pts = gauss-legendre

[solver-interfaces-quad]
flux-pts = gauss-legendre

[solver-elements-quad]
soln-pts = gauss-legendre

[soln-bcs-wall]
type = {wall}

[soln-ics]
rho = 1 + 0.2*exp(-60*((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5)))
u = 0
v = 0
p = 1 + 0.2*exp(-60*((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5)))

[soln-plugin-nancheck]
nsteps = 10

[soln-plugin-writer]
dt-out = 0.1
basedir = .
basename = {system}-p{order}-{{n:03d}}
"""

WALLS = {'euler': 'slp-adia-wall', 'navier-stokes': 'no-slp-adia-wall'}

# (mean, min, max) of each variable at t = 0.1, measured with the CUDA
# backend on a real GeForce RTX 3060 (sm_86)
REF = {
    'euler-p3': {
        'rho': (1.0104798245621365, 0.9999999969570285, 1.0624569870082898),
        'E': (2.526239730422179, 2.4999999874730183, 2.6318221309211207),
        'rhou': (None, -0.05777332826947586, 0.05777332826947652),
    },
    'euler-p6': {
        'rho': (1.0104803297429528, 1.0000000002477085, 1.0617373815937838),
        'E': (2.526245388127024, 2.500000000866923, 2.6306541802669083),
        'rhou': (None, -0.05942859840404627, 0.059428598404052804),
    },
    'navier-stokes-p3': {
        'rho': (1.01047418981177, 1.00000002026884, 1.07090763729731),
        'E': (2.52619685697675, 2.50000009567184, 2.62232340286952),
        'rhou': (2.7541229757374e-07, -0.0537879294887166,
                 0.0541908828478559),
    },
}


def mesh(out):
    n = NCELL
    nid = lambda i, j: j*(n + 1) + i + 1
    nodes = [(i/n, j/n) for j in range(n + 1) for i in range(n + 1)]
    quads = [(nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1))
             for j in range(n) for i in range(n)]
    lines = ([(nid(i, 0), nid(i + 1, 0)) for i in range(n)] +
             [(nid(n, j), nid(n, j + 1)) for j in range(n)] +
             [(nid(i + 1, n), nid(i, n)) for i in range(n)] +
             [(nid(0, j + 1), nid(0, j)) for j in range(n)])

    with open(out, 'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        f.write('$PhysicalNames\n2\n1 1 "wall"\n2 2 "fluid"\n'
                '$EndPhysicalNames\n')
        f.write(f'$Nodes\n{len(nodes)}\n')
        for k, (x, y) in enumerate(nodes, 1):
            f.write(f'{k} {x} {y} 0\n')
        f.write('$EndNodes\n')
        f.write(f'$Elements\n{len(lines) + len(quads)}\n')
        for k, (a, b) in enumerate(lines, 1):
            f.write(f'{k} 1 2 1 1 {a} {b}\n')
        for k, q in enumerate(quads, len(lines) + 1):
            f.write(f'{k} 3 2 2 2 {q[0]} {q[1]} {q[2]} {q[3]}\n')
        f.write('$EndElements\n')


def config(system, order, out):
    with open(out, 'w') as f:
        f.write(CONFIG.format(system=system, order=order,
                              wall=WALLS[system]))


def check(fname, key):
    with h5py.File(fname) as f:
        (name,) = f['soln']
        soln = f['soln'][name][()]['soln']

    ok = True
    for var in ('rho', 'rhou', 'rhov', 'E'):
        x = soln[var]
        print(f'{var:5s} mean {x.mean():.15g} min {x.min():.15g} '
              f'max {x.max():.15g}')
        if not np.isfinite(x).all():
            print(f'  FAIL: non-finite values in {var}')
            ok = False

    # The closed, symmetric box keeps the net Euler momentum at zero
    if (key.startswith('euler') and (abs(soln['rhou'].mean()) > 1e-12 or
                                     abs(soln['rhov'].mean()) > 1e-12)):
        print('  FAIL: net momentum is not zero')
        ok = False

    for var, (mean, lo, hi) in REF.get(key, {}).items():
        x = soln[var]
        for what, got, ref in (('mean', x.mean(), mean), ('min', x.min(), lo),
                               ('max', x.max(), hi)):
            if ref is not None and not np.isclose(got, ref, rtol=1e-9,
                                                  atol=1e-12):
                print(f'  FAIL: {var} {what} {got!r} != {ref!r}')
                ok = False

    return ok


if __name__ == '__main__':
    cmd, *args = sys.argv[1:]

    if cmd == 'mesh':
        mesh(*args)
    elif cmd == 'config':
        config(*args)
    elif cmd == 'check':
        sys.exit(0 if check(*args) else 1)
