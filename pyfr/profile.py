"""Fluent 风格的 profile 支持：从 CSV 点集文件构建入口边界条件。

CSV 文件包含一系列空间点及其物理量（列名不区分大小写）：
    x, y, z   —— 坐标（2D 时 z 列可省略）
    u, v, w   —— 速度分量（2D 时 w 列可省略）
    P         —— 静压
    T         —— 静温

使用方法（在配置文件中）：
    [soln-bcs-inlet]
    type = sub-in-profile
    file = inlet-profile.csv
    # 可选: interp = idw | linear | nearest（默认 idw）
    #   idw:    k-近邻逆距离加权
    #   linear: 结构化网格双线性/线性插值 + 端点钳制（Fluent profile 等价）
    #   nearest: 最近邻
    # 可选: power = 2（IDW 幂次）
    # 可选: knn = 8（最近邻个数）
"""

import csv

import numpy as np


class Profile:
    # 依据网格维度确定必需与可选的列
    _coord_cols = ('x', 'y', 'z')
    _vel_cols = ('u', 'v', 'w')
    _scal_cols = ('P', 'T')

    def __init__(self, pts, pri, ndims):
        # pts: (n, ndims) 坐标；pri: (n, ndims+2) 物理量 [u..w, P, T]
        self.pts = pts
        self.pri = pri
        self.ndims = ndims

    @classmethod
    def from_csv(cls, fname, ndims):
        # 读取并解析 CSV 文件
        with open(fname, newline='') as f:
            rows = list(csv.reader(f))

        # 去除空行
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            raise ValueError(f'Profile 文件为空: {fname!r}')

        # 判断首行是否为表头（若包含非数值字段则视为表头）
        def _isnum(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        if all(_isnum(c) for c in rows[0] if c.strip()):
            header = None
            data = rows
        else:
            header = [c.strip().lower() for c in rows[0]]
            data = rows[1:]

        # 确定列映射（表头统一小写后匹配）
        if header:
            cols = {c: i for i, c in enumerate(header)}
        else:
            # 无表头时按约定顺序: x,y,z,u,v,w,p,t
            names = [c.lower() for c in cls._coord_cols + cls._vel_cols +
                     cls._scal_cols]
            cols = {c: i for i, c in enumerate(names)}

        # 组装必需列名（2D 无 z/w）
        required = (cls._coord_cols[:ndims] + cls._vel_cols[:ndims] +
                    cls._scal_cols)
        missing = [c for c in required if c.lower() not in cols]
        if missing:
            raise ValueError(f'Profile 文件 {fname!r} 缺少列: {missing} '
                             f'(表头: {header})')

        # 提取数值（跳过无法解析的行）
        coords, scalars = [], []
        for r in data:
            try:
                coords.append([float(r[cols[c]]) for c in
                               cls._coord_cols[:ndims]])
                scalars.append([float(r[cols[c.lower()]]) for c in
                                cls._vel_cols[:ndims]] +
                               [float(r[cols['p']]), float(r[cols['t']])])
            except (ValueError, IndexError):
                continue

        if not coords:
            raise ValueError(f'Profile 文件 {fname!r} 中无有效数据行')

        return cls(np.asarray(coords), np.asarray(scalars), ndims)

    @property
    def npoints(self):
        return len(self.pts)

    def summary(self):
        # 用于运行信息输出：各物理量的取值范围
        nd = self.ndims
        umag = np.sqrt(np.sum(self.pri[:, :nd]**2, axis=1))
        P, T = self.pri[:, nd], self.pri[:, nd + 1]

        sp = lambda a: (float(np.min(a)), float(np.max(a)))

        return (f'{self.npoints} 点, |u|∈({sp(umag)[0]:.4g}, '
                f'{sp(umag)[1]:.4g}), P∈({sp(P)[0]:.4g}, {sp(P)[1]:.4g}), '
                f'T∈({sp(T)[0]:.4g}, {sp(T)[1]:.4g})')

    def _interp_chunk(self, chunk):
        # 逆距离加权（IDW）/ 线性 / 最近邻处理
        method, power, knn = self.method, self.power, self.knn

        if method == 'linear':
            return self._interp_linear(chunk)

        # 与所有 CSV 点的平方距离: (m, n)
        d2 = (np.sum(chunk*chunk, axis=1, keepdims=True) +
              np.sum(self.pts*self.pts, axis=1) -
              2*chunk @ self.pts.T)
        d2 = np.maximum(d2, 0)

        if method == 'nearest':
            idx = np.argmin(d2, axis=1)
            return self.pri[idx]

        # knn 最近邻
        knn = min(knn, self.pri.shape[0])
        idx = np.argpartition(d2, knn - 1, axis=1)[:, :knn]

        # 取每行 knn 个最近邻的距离
        dk = np.take_along_axis(d2, idx, axis=1)
        wk = 1.0/np.maximum(dk, 1e-20)**power
        wk /= wk.sum(axis=1, keepdims=True)

        return np.einsum('ij,ijk->ik', wk, self.pri[idx])

    def _interp_linear(self, chunk):
        # Fluent 式线性插值: 结构化网格双线性, 端点钳制
        # lazy 构建 1D 坐标轴 + 2D 网格表
        m, n = chunk.shape[0], self.pts.shape[0]
        if not hasattr(self, '_lin_y'):
            yv = np.unique(np.round(self.pts[:, 1], 8))
            zv = np.unique(np.round(self.pts[:, 2], 8)) if self.ndims == 3 else np.array([0.0])
            # 排序并去重
            yv = np.sort(yv)
            zv = np.sort(zv)
            self._lin_y, self._lin_z = yv, zv
            # 建 (ny, nz) 索引表: 每个 (yi,zi) 的 pri 均值
            ny, nz = len(yv), len(zv)
            self._lin_ny, self._lin_nz = ny, nz
            tab = np.zeros((ny, nz, self.pri.shape[1]))
            cnt = np.zeros((ny, nz))
            # 坐标 -> 索引
            y2i = {round(float(v), 8): i for i, v in enumerate(yv)}
            z2i = {round(float(v), 8): i for i, v in enumerate(zv)}
            for pt, pr in zip(self.pts, self.pri):
                yi = y2i.get(round(float(pt[1]), 8))
                zi = z2i.get(round(float(pt[2]), 8)) if self.ndims == 3 else 0
                if yi is None or zi is None:
                    continue
                tab[yi, zi] += pr
                cnt[yi, zi] += 1
            cnt = np.maximum(cnt, 1)
            self._lin_tab = tab / cnt[..., None]

        yv, zv, tab = self._lin_y, self._lin_z, self._lin_tab
        out = np.empty((m, self.pri.shape[1]))

        for r, pt in enumerate(chunk):
            yq = pt[1]
            # y 插值索引 (钳制到端点)
            iy1 = int(np.searchsorted(yv, yq))
            if iy1 <= 0:
                iy0, iy1 = 0, 0
                wy = 0.0
            elif iy1 >= len(yv):
                iy0, iy1 = len(yv) - 1, len(yv) - 1
                wy = 0.0
            else:
                iy0 = iy1 - 1
                den = yv[iy1] - yv[iy0]
                wy = (yq - yv[iy0]) / den if den > 1e-14 else 0.0

            if self.ndims == 3:
                zq = pt[2]
                iz1 = int(np.searchsorted(zv, zq))
                if iz1 <= 0:
                    iz0, iz1 = 0, 0
                    wz = 0.0
                elif iz1 >= len(zv):
                    iz0, iz1 = len(zv) - 1, len(zv) - 1
                    wz = 0.0
                else:
                    iz0 = iz1 - 1
                    den = zv[iz1] - zv[iz0]
                    wz = (zq - zv[iz0]) / den if den > 1e-14 else 0.0

                v00 = tab[iy0, iz0]
                v01 = tab[iy0, iz1]
                v10 = tab[iy1, iz0]
                v11 = tab[iy1, iz1]
                lo = (1 - wz) * v00 + wz * v01
                hi = (1 - wz) * v10 + wz * v11
                out[r] = (1 - wy) * lo + wy * hi
            else:
                iz0 = 0
                v0 = tab[iy0, iz0]
                v1 = tab[iy1, iz0]
                out[r] = (1 - wy) * v0 + wy * v1

        return out

    def interpolate(self, plocs, method='idw', power=2.0, knn=8, csize=4096):
        # plocs: (m, ndims) 目标点坐标；返回 (m, 5)
        self.method, self.power, self.knn = method, power, knn

        out = np.empty((len(plocs), self.pri.shape[1]))
        for s in range(0, len(plocs), csize):
            e = min(s + csize, len(plocs))
            out[s:e] = self._interp_chunk(plocs[s:e])

        return out


class ProfileBCMixin:
    """profile 边界条件公共逻辑（供 Euler / Navier-Stokes 使用）。

    约定：配置节中提供
        file      —— CSV profile 文件路径（必需）
        R         —— 气体常数（可选；默认 cp*(gamma-1)/gamma，
                     其中 cp、gamma 来自 [constants]）
        interp    —— 插值方式 idw / nearest（可选）
        power     —— IDW 幂次（可选）
        knn       —— 最近邻个数（可选）
    """
    _profile_nvars = None  # 根据 ndims 在运行时设置为 ndims + 2

    def _setup_profile(self, lhs):
        from pyfr.profile import Profile

        cfg, sect = self.cfg, self.cfgsect

        # 加载 CSV profile
        fname = cfg.getpath(sect, 'file')
        self.profile = Profile.from_csv(fname, self.ndims)

        # 物理量顺序: u, v, w(3D), P, T
        raw = self.profile.pri

        if np.any(raw[:, self.ndims + 1] <= 0) or np.any(raw[:, self.ndims] <= 0):
            raise ValueError(f'Profile 文件 {fname!r} 中存在非正的 P 或 T')

        # 气体常数
        consts = cfg.items_as('constants', float)
        if 'R' in consts:
            R = consts['R']
        elif 'cp' in consts:
            R = consts['cp']*(consts['gamma'] - 1)/consts['gamma']
        else:
            R = cfg.getfloat(sect, 'R')

        # 获取边界面上所有通量点的物理坐标（与 _const_mat 相同的顺序）
        from pyfr.solvers.base.inters import _get_inter_arrays
        m = _get_inter_arrays(lhs, 'get_ploc_for_inters', self.elemap,
                              self._perm)
        # m[0] 形状为 (ninterfpts, ndims)
        plocs = m[0] if m else np.empty((0, self.ndims))

        # 插值到面通量点
        interp = cfg.get(sect, 'interp', 'idw')
        power = cfg.getfloat(sect, 'power', 2.0)
        knn = cfg.getint(sect, 'knn', 8)
        vals = self.profile.interpolate(plocs, method=interp, power=power,
                                        knn=knn)

        # 插值结果: [u, v, w, P, T] -> 转成 [rho, u, v, w, p]
        uvw = vals[:, :self.ndims]
        p = vals[:, self.ndims]
        t = vals[:, self.ndims + 1]
        rho = p/(R*t)
        states = np.column_stack([rho, uvw, p])

        # 通过外部常量矩阵传入内核: 形状 (ndims+2, ninterfpts)
        self._profile_nvars = self.ndims + 2
        cm = self._be.const_matrix(np.atleast_2d(states.T))
        self.set_external('profile', f'in fpdtype_t[{self._profile_nvars}]',
                          value=cm)

    def profile_summary(self):
        return self.profile.summary()
