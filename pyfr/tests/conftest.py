import json


def pytest_addoption(parser):
    group = parser.getgroup('luinv', 'LU inverses of the block-Jacobi '
                            'preconditioner (test_luinv.py)')
    group.addoption('--luinv-full', action='store_true',
                    help='run the whole validation matrix')
    group.addoption('--luinv-bench', action='store_true',
                    help='benchmark the HIP kernels against Mako, and run '
                    'nothing else')
    group.addoption('--luinv-bench-report', metavar='FILE',
                    help='also write the benchmark to FILE as JSON lines; '
                    'implies --luinv-bench')


def _bench(config):
    return (config.getoption('luinv_bench') or
            config.getoption('luinv_bench_report') is not None)


def pytest_configure(config):
    config.addinivalue_line('markers', 'full: in the whole validation '
                            'matrix of the LU inverses (--luinv-full)')
    config.addinivalue_line('markers', 'bench: in the benchmark of the LU '
                            'inverses (--luinv-bench)')

    # Rows of the benchmark, which the tests append to
    config.luinv_bench = []


def pytest_collection_modifyitems(config, items):
    full, bench = config.getoption('luinv_full'), _bench(config)

    # The benchmark on its own, or every test but it
    keep, drop = [], []
    for item in items:
        if bench:
            wanted = item.get_closest_marker('bench')
        else:
            wanted = not (item.get_closest_marker('bench') or
                          (item.get_closest_marker('full') and not full))

        (keep if wanted else drop).append(item)

    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_terminal_summary(terminalreporter, config):
    rows = config.luinv_bench
    if not rows:
        return

    tr = terminalreporter
    tr.section('LU inverse benchmark: HIP vs Mako')
    tr.write_line(f'{"gpu":<8} {"n":>5} {"precision":<9} {"matrices":>8} '
                  f'{"mako ms":>9} {"hip ms":>9} {"speedup":>8}')
    for r in rows:
        tr.write_line(f'{r["gpu"].split(":")[0]:<8} {r["n"]:>5} '
                      f'{r["precision"]:<9} {r["neles"]:>8} '
                      f'{r["mako_ms"]:>9.3f} {r["hip_ms"]:>9.3f} '
                      f'{r["speedup"]:>7.2f}x')

    if path := config.getoption('luinv_bench_report'):
        with open(path, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')

        tr.write_line(f'Written to {path}')
