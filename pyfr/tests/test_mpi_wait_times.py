import gc
from types import SimpleNamespace

import pytest

import pyfr.backends.base.types as btypes
from pyfr.backends.base.types import Graph, XchgMatrix
from pyfr.solvers.base.system import BaseSystem


class _Cfg:
    def __init__(self, collect=True, per_peer=True, n=100):
        self.collect = collect
        self.per_peer = per_peer
        self.n = n

    def getbool(self, section, option, default=False):
        return {
            'collect-wait-times': self.collect,
            'collect-wait-times-per-peer': self.per_peer,
        }.get(option, default)

    def getint(self, section, option, default=None):
        return self.n if option == 'collect-wait-times-len' else default


class _Req:
    pass


def test_xchg_matrix_records_request_metadata(monkeypatch):
    backend = SimpleNamespace(cfg=_Cfg())
    mat = XchgMatrix.__new__(XchgMatrix)
    mat.backend = backend
    mat.hdata = SimpleNamespace(nbytes=128)

    class Comm:
        def Recv_init(self, data, pid, tag):
            return _Req()

        def Send_init(self, data, pid, tag):
            return _Req()

    monkeypatch.setattr(btypes, 'autofree', lambda req: req)

    recv = mat.recvreq(Comm(), 2, 10)
    send = mat.sendreq(Comm(), 3, 11)

    rid_recv, rid_send = id(recv), id(send)
    assert backend.mpi_req_info[rid_recv] == (2, False, 128)
    assert backend.mpi_req_info[rid_send] == (3, True, 128)

    del recv, send
    gc.collect()
    assert rid_recv not in backend.mpi_req_info
    assert rid_send not in backend.mpi_req_info


def test_per_peer_tracking_requires_wait_tracking():
    backend = SimpleNamespace(cfg=_Cfg(collect=False, per_peer=True))
    with pytest.raises(ValueError, match='requires collect-wait-times'):
        Graph(backend)


def test_detailed_wait_attribution_conserves_time(monkeypatch):
    reqs = [_Req() for _ in range(4)]
    backend = SimpleNamespace(
        cfg=_Cfg(),
        mpi_req_info={
            id(reqs[0]): (1, True, 10),
            id(reqs[1]): (2, False, 20),
            id(reqs[2]): (3, False, 30),
            id(reqs[3]): (2, True, 40),
        },
    )
    batches = iter([[0, 2], [1], [3], None])

    class Prequest:
        Startall = staticmethod(lambda reqs: None)
        Waitall = staticmethod(lambda reqs: None)
        Waitsome = staticmethod(lambda reqs: next(batches))

    fake_mpi = SimpleNamespace(Prequest=Prequest, REQUEST_NULL=object())
    ticks = iter([0, 100, 100, 400, 400, 450, 450, 450])
    monkeypatch.setattr(btypes, 'mpi', fake_mpi)
    monkeypatch.setattr(btypes.time, 'perf_counter_ns', lambda: next(ticks))

    graph = Graph(backend)
    graph.mpi_reqs = reqs
    graph._waitall(reqs)

    send, recv = graph.get_wait_times_by_peer()

    assert graph.get_wait_times() == pytest.approx([450e-9])
    assert send[1] == pytest.approx([50e-9])
    assert send[2] == pytest.approx([50e-9])
    assert recv[2] == pytest.approx([300e-9])
    assert recv[3] == pytest.approx([50e-9])

    attributed = sum(map(sum, send.values())) + sum(map(sum, recv.values()))
    assert attributed == pytest.approx(graph.get_wait_times()[0])
    assert graph.get_mpi_bytes() == ({1: 10, 2: 40}, {2: 20, 3: 30})


class _FakeGraph:
    def __init__(self, send=None, recv=None, nbytes=None):
        self._send = send or {}
        self._recv = recv or {}
        self._nbytes = nbytes or ({}, {})

    def get_wait_times_by_peer(self):
        return self._send, self._recv

    def get_mpi_bytes(self):
        return self._nbytes


class _FakeSystem:
    _rhs_uin_fout = {(0, 1), (2, 2)}

    def __init__(self, graphs):
        self.graphs = graphs

    def _rhs_graphs(self, uinbank, foutbank):
        return self.graphs[uinbank, foutbank]


def test_rhs_peer_stats_pool_equivalent_graphs():
    nb0 = ({1: 64}, {1: 64})
    nb1 = ({2: 96}, {2: 96})
    graphs = {
        (0, 1): (
            _FakeGraph({1: [1.0, 3.0]}, {2: [2.0]}, nb0),
            _FakeGraph({2: [4.0]}, {3: [8.0]}, nb1),
            _FakeGraph(),
        ),
        (2, 2): (
            _FakeGraph({1: [5.0]}, {2: [4.0]}, nb0),
            _FakeGraph({2: [6.0]}, {3: [10.0]}, nb1),
            _FakeGraph(),
        ),
    }
    system = _FakeSystem(graphs)

    stats = BaseSystem.rhs_wait_times_by_peer(system)
    assert len(stats) == 3

    send0, recv0 = stats[0]
    assert send0[1] == pytest.approx((3.0, 2.0, 3.0))
    assert recv0[2][0] == pytest.approx(3.0)
    assert recv0[2][2] == pytest.approx(3.0)

    send1, recv1 = stats[1]
    assert send1[2][0] == pytest.approx(5.0)
    assert send1[2][2] == pytest.approx(5.0)
    assert recv1[3][0] == pytest.approx(9.0)
    assert recv1[3][2] == pytest.approx(9.0)

    assert stats[2] == ({}, {})
    assert BaseSystem.rhs_mpi_bytes(system) == [nb0, nb1, ({}, {})]


def test_rhs_mpi_bytes_reject_inconsistent_graph_variants():
    graphs = {
        (0, 1): (_FakeGraph(nbytes=({1: 64}, {1: 64})),),
        (2, 2): (_FakeGraph(nbytes=({1: 128}, {1: 64})),),
    }
    system = _FakeSystem(graphs)

    with pytest.raises(RuntimeError, match='Inconsistent MPI request sizes'):
        BaseSystem.rhs_mpi_bytes(system)


def test_regular_wait_tracking_uses_waitall(monkeypatch):
    backend = SimpleNamespace(cfg=_Cfg(collect=True, per_peer=False))
    calls = []

    class Prequest:
        Startall = staticmethod(lambda reqs: None)
        Waitall = staticmethod(lambda reqs: calls.append(list(reqs)))

    fake_mpi = SimpleNamespace(Prequest=Prequest)
    ticks = iter([10, 60])
    monkeypatch.setattr(btypes, 'mpi', fake_mpi)
    monkeypatch.setattr(btypes.time, 'perf_counter_ns', lambda: next(ticks))

    reqs = [_Req(), _Req()]
    graph = Graph(backend)
    graph._waitall(reqs)

    assert calls == [reqs]
    assert graph.get_wait_times() == pytest.approx([50e-9])
    assert graph.get_wait_times_by_peer() == ({}, {})
