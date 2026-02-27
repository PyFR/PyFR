import numpy as np
from pyfr.partitioners.metis import METISPartitioner
from pyfr.regions import parse_region_expr

class ManualPartitioner(METISPartitioner):
    name = 'manual'

    def _get_element_centroids(self):
        all_centroids = []
        # BasePartitioner로부터 상속받은 self._elements 활용
        for etype, eles in self._elements.items():
            # spt 데이터에서 노드 방향(axis=0)으로 평균을 내어 중심점 계산
            all_centroids.append(np.mean(eles, axis=0))
        return np.vstack(all_centroids)

    def _partition_graph(self, graph, partwts):
        # 1. 먼저 부모 클래스(METIS)의 표준 파티셔닝 결과를 받습니다.
        # 이 결과는 격자 개수에 맞춰 예쁘게 4개로 나뉜 상태입니다.
        parts = super()._partition_graph(graph, partwts)
        
        # 2. --popt regions:"..."로 들어온 리전 정보를 파싱합니다.
        region_raw = self.opts.get('regions', '')
        if not region_raw:
            return parts # 리전 정보가 없으면 일반 METIS 결과 반환

        centroids = self._get_element_centroids()
        region_exprs = region_raw.split(';')

        # 3. 지정된 리전들에 대해 강제로 랭크를 고정합니다.
        # 예: 첫 번째 리전은 Rank 0, 두 번째 리전은 Rank 1...
        for r_idx, expr in enumerate(region_exprs):
            if r_idx >= len(partwts):
                break
                
            reg = parse_region_expr(expr.strip())
            mask = reg.pts_in_region(centroids)
            
            # 해당 좌표 영역에 속하는 요소들의 랭크를 강제로 고정
            parts[mask] = r_idx

        return parts.astype(np.int32)