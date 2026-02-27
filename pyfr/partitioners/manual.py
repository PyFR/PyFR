"""
pyfr/partitioners/manual.py

Assigns each specified region to a separate partition, and automatically 
partitions the remaining elements using METIS.
If the number of regions == nparts, the partitioning is handled purely 
manually without METIS.

Element assignment is determined based on the element's centroid.
This ensures that elements at adjacent box boundaries do not overlap 
and are divided cleanly.

Usage
-----
# [Mode 1] regions: Specify region expressions directly
#   1 region -> partition 0, remaining 3 partitions handled by METIS
pyfr partition add mesh.pyfrm 4 myname -p manual \\
    --popt 'regions:box((0,0,0),(0.1,0.03,0.0075))'

#   4 regions -> partitions 0~3, no METIS (pure manual)
pyfr partition add mesh.pyfrm 4 myname -p manual \\
    --popt 'regions:box((0,0,-0.015),(0.1,0.03,-0.0075));box((0,0,-0.0075),(0.1,0.03,0));box((0,0,0),(0.1,0.03,0.0075));box((0,0,0.0075),(0.1,0.03,0.015))'

# [Mode 2] split_axis: Divide into nparts equally based on centroid quantiles
#   Divide into 4 along the z-axis (each partition has roughly equal element counts)
pyfr partition add mesh.pyfrm 4 myname -p manual \\
    --popt split_axis:z

#   Divide into 8 along the x-axis
pyfr partition add mesh.pyfrm 8 myname -p manual \\
    --popt split_axis:x

Rules (regions mode)
------------------
- Use ';' to separate region expressions in the 'regions' option.
- First region -> partition 0, second -> partition 1, etc.
- If number of regions < nparts: Remaining elements -> METIS (partitions N to nparts-1).
- If number of regions == nparts: Pure manual.
- If number of regions > nparts: Raises an error.
- Centroid-based assignment ensures no overlaps at boundaries.

Rules (split_axis mode)
---------------------
- Equally divides the mesh into nparts quantiles based on the specified centroid axis.
- Ensures each partition has a nearly identical number of elements.
- Cannot be used simultaneously with the 'regions' option.
"""

import numpy as np

from pyfr.partitioners.base import BasePartitioner, Graph


class ManualPartitioner(BasePartitioner):
    name = 'manual'
    has_part_weights = True
    has_multiple_constraints = True

    int_opts = set()
    enum_opts = {}
    dflt_opts = {}

    def __init__(self, partwts, elewts=None, nsubeles=64, opts={}):
        opts = dict(opts)

        self._regions_str  = opts.pop('regions', None)
        self._split_axis   = opts.pop('split_axis', None)

        # Cannot use both options simultaneously
        if self._regions_str and self._split_axis:
            raise ValueError(
                'Cannot use both "regions" and "split_axis" at the same time.'
            )

        if not self._regions_str and not self._split_axis:
            raise ValueError(
                'ManualPartitioner requires either:\n'
                '  --popt \'regions:box(...)\'\n'
                '  --popt split_axis:x  (or y, z)'
            )

        if len(partwts) < 2:
            raise ValueError('ManualPartitioner requires at least 2 partitions')

        # Regions mode: Parse expressions
        if self._regions_str:
            self._region_exprs = [r.strip() for r in self._regions_str.split(';')
                                  if r.strip()]
            n_manual = len(self._region_exprs)
            if n_manual > len(partwts):
                raise ValueError(
                    f'{n_manual} region(s) specified but only {len(partwts)} '
                    f'total partitions. Need at least {n_manual} partitions.'
                )
        else:
            self._region_exprs = []
            if self._split_axis not in ('x', 'y', 'z'):
                raise ValueError(
                    f'Invalid split_axis "{self._split_axis}". '
                    'Must be one of: x, y, z'
                )

        super().__init__(partwts, elewts=elewts, nsubeles=nsubeles, opts=opts)

    def partition(self, mesh, progress=None):
        from pyfr.progress import NullProgressSequence
        from pyfr.regions import parse_region_expr

        if progress is None:
            progress = NullProgressSequence()

        # 1. Global connectivity ------------------------------------------
        with progress.start('Construct global connectivity array'):
            con, ecurved, edisps, cdisps = self.construct_global_con(mesh)

        total_eles = int(ecurved.shape[0])
        elewts_fn  = self._get_elewts_fn(edisps)

        # 2. Periodic element merging -------------------------------------
        with progress.start('Group periodic elements'):
            pmcon, exwts, pmerge = self._group_periodic_eles(
                mesh, con, cdisps, elewts_fn)

        # 3. Construct dual graph -----------------------------------------
        with progress.start('Construct graph'):
            full_graph, vemap = self._construct_graph(pmcon, elewts_fn,
                                                      exwts=exwts)

        # 4. Compute centroids --------------------------------------------
        with progress.start('Compute element centroids'):
            all_nodes    = mesh['nodes']['location'][()]
            centroids    = []
            for etype in sorted(edisps):
                nidxs = mesh['eles'][etype]['nodes'][()]
                centroids.append(all_nodes[nidxs].mean(axis=1))
            all_centroids = np.vstack(centroids)   # (total_eles, ndim)

        # 5. Calculate partition assignments (vparts_v) -------------------
        if self._split_axis:
            vparts_v = self._assign_by_split_axis(
                all_centroids, vemap, progress)
        else:
            vparts_v = self._assign_by_regions(
                all_centroids, total_eles, edisps, vemap,
                full_graph, mesh, progress, parse_region_expr)

        # 6. Ungroup periodic elements ------------------------------------
        with progress.start('Ungroup periodic elements'):
            vparts = self._ungroup_periodic_eles(pmerge, vemap, vparts_v)

        # 7. Sanity check -------------------------------------------------
        unique_parts = np.unique(vparts)
        if len(unique_parts) != self.nparts:
            raise RuntimeError(
                f'Expected {self.nparts} partitions, got '
                f'{len(unique_parts)}: {unique_parts.tolist()}.'
            )

        # 8. Construct canonical partitioning structure -------------------
        with progress.start('Construct partitioning'):
            pinfo = self.construct_partitioning(
                mesh, ecurved, edisps, con, vparts)

        return pinfo

    # ------------------------------------------------------------------
    # split_axis mode: Equal division based on centroid quantiles
    # ------------------------------------------------------------------
    def _assign_by_split_axis(self, all_centroids, vemap, progress):
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[self._split_axis]
        nparts   = self.nparts

        with progress.start(
                f'Split axis={self._split_axis}, {nparts} equal parts '
                f'(quantile-based)'):

            # Use centroid axis values based on total_eles
            # (Using vemap directly might omit elements from periodic merging)
            cvals_all = all_centroids[:, axis_idx]   # (total_eles,)

            # Calculate internal boundaries (nparts-1): e.g., 25%, 50%, 75%
            quantiles  = np.linspace(0, 100, nparts + 1)[1:-1]
            inner_boundaries = np.percentile(cvals_all, quantiles)

            # Assign partition numbers to all elements
            vparts_full = np.searchsorted(inner_boundaries, cvals_all,
                                          side='right').astype(np.int32)

            # Boundaries for display purposes (including +-inf)
            boundaries = np.concatenate(([-np.inf], inner_boundaries, [np.inf]))

            # Print element distribution info
            for p in range(nparts):
                n    = int((vparts_full == p).sum())
                lo_s = f'{boundaries[p]:.6f}'   if np.isfinite(boundaries[p])   else '-inf'
                hi_s = f'{boundaries[p+1]:.6f}' if np.isfinite(boundaries[p+1]) else '+inf'
                print(f'[manual partitioner] partition {p}: '
                      f'{n} elements  '
                      f'({self._split_axis}=[{lo_s}, {hi_s}])')

            # Map back via vemap (reflecting periodic merging)
            vparts_v = vparts_full[vemap]

        return vparts_v

    # ------------------------------------------------------------------
    # regions mode: Centroid-based region identification + METIS
    # ------------------------------------------------------------------
    def _assign_by_regions(self, all_centroids, total_eles, edisps, vemap,
                           full_graph, mesh, progress, parse_region_expr):
        n_manual = len(self._region_exprs)
        n_auto   = self.nparts - n_manual
        rdata    = mesh['regions'] if 'regions' in mesh else None

        # Collect centroid masks for each region
        masks_v = []
        for part_id, expr in enumerate(self._region_exprs):
            with progress.start(f'Identify region {part_id}: {expr}'):
                region_obj = parse_region_expr(expr, rdata)
                inside     = region_obj.pts_in_region(all_centroids)
                mask_full  = inside.astype(bool)

            mask_v = mask_full[vemap]
            if not mask_v.any():
                raise ValueError(
                    f'Region {part_id} ("{expr}") contains no elements '
                    f'(centroid-based). Check expression and mesh extents.'
                )
            masks_v.append(mask_v)

        # Overlap warnings
        for i in range(len(masks_v)):
            for j in range(i + 1, len(masks_v)):
                n_overlap = int((masks_v[i] & masks_v[j]).sum())
                if n_overlap > 0:
                    print(
                        f'[manual partitioner] WARNING: '
                        f'region {i} ("{self._region_exprs[i]}") and '
                        f'region {j} ("{self._region_exprs[j]}") overlap '
                        f'on {n_overlap} element(s). '
                        f'Assigning to region {i} (first region wins).'
                    )

        # Assign partition numbers
        vparts_v = np.full(len(vemap), -1, dtype=np.int32)
        for part_id, mask_v in enumerate(masks_v):
            vparts_v[(vparts_v == -1) & mask_v] = part_id

        # Handle unassigned elements
        rest_mask_v = (vparts_v == -1)
        n_rest = int(rest_mask_v.sum())

        if n_rest > 0 and n_auto == 0:
            raise RuntimeError(
                f'{n_rest} element(s) not covered by any region, '
                f'but no METIS partitions available (n_regions == nparts). '
                f'Adjust region expressions or increase nparts.'
            )

        if n_rest > 0:
            with progress.start(
                    f'Build sub-graph ({n_rest} non-region elements)'):
                nv  = len(vemap)
                g2s = np.full(nv, -1, dtype=np.int64)
                g2s[np.where(rest_mask_v)[0]] = np.arange(n_rest, dtype=np.int64)

                vtab_full = full_graph.vtab
                etab_full = full_graph.etab

                adj = [[] for _ in range(n_rest)]
                for vi in range(nv):
                    si = int(g2s[vi])
                    if si < 0:
                        continue
                    for ei in range(int(vtab_full[vi]), int(vtab_full[vi + 1])):
                        sj = int(g2s[int(etab_full[ei])])
                        if sj >= 0:
                            adj[si].append(sj)

                vtab = np.zeros(n_rest + 1, dtype=np.int64)
                for i, nb in enumerate(adj):
                    vtab[i + 1] = vtab[i] + len(nb)
                etab = np.empty(int(vtab[-1]), dtype=np.int64)
                for i, nb in enumerate(adj):
                    s = int(vtab[i])
                    for j, v in enumerate(nb):
                        etab[s + j] = v

                sub_graph = Graph(vtab, etab,
                                  np.ones((n_rest, 1), dtype=np.int32),
                                  np.ones(len(etab), dtype=np.int32))

            with progress.start(f'METIS: partition into {n_auto} sub-parts'):
                from pyfr.partitioners.metis import METISPartitioner
                sub_partwts = [1] * n_auto
                metis = METISPartitioner.__new__(METISPartitioner)
                METISPartitioner.__init__(
                    metis, sub_partwts,
                    elewts={et: 1 for et in edisps}
                )
                sub_vparts = metis._partition_graph(sub_graph, sub_partwts)
                vparts_v[rest_mask_v] = sub_vparts.astype(np.int32) + n_manual

        return vparts_v