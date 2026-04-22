"""
BestIns: greedy insertion heuristic (Section 6.3).

Thuật toán:
1. Dùng ORTools plan routes cho static customers (b_j == 0)
2. Khi dynamic customer xuất hiện:
   - Thử insert vào mọi vị trí trong routes
   - Chọn vị trí có insertion cost (detour) nhỏ nhất
   - Bỏ qua nếu insertion cost > threshold
"""
import numpy as np
from ..env.instance import Instance
from ..env.state import *
from .ortools_solver import _build_ortools_model


class BestInsBaseline:

    def __init__(self, insertion_threshold: float = 5.0):
        """
        Args:
            insertion_threshold: bỏ qua customer nếu detour > threshold (km)
        """
        self.threshold = insertion_threshold

    def solve_online(self, instance: Instance) -> dict:
        """Simulate online execution với BestIns.
        
        Returns:
            total_cost, qos
        """
        n = instance.n_customers
        m = instance.n_vehicles
        all_xy = np.vstack([instance.depot_xy, instance.cust_xy])

        # Khởi tạo routes chỉ với static customers
        static_mask = instance.cust_b == 0
        static_idx = [j + 1 for j in range(n) if static_mask[j]]  # 1-indexed

        # Build initial plan with ORTools for static customers only
        # (simplified: dùng nearest neighbor thay ORTools cho đơn giản)
        routes = self._nearest_neighbor_init(instance, static_idx)

        # Simulate: process dynamic customers theo thứ tự appear time
        dyn_customers = sorted(
            [j + 1 for j in range(n) if not static_mask[j]],
            key=lambda j: instance.cust_b[j - 1]
        )

        for j in dyn_customers:
            best_pos, best_route, best_cost = None, None, float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):  # insert between route[pos-1] and route[pos]
                    # Insertion cost = detour
                    prev = route[pos - 1]
                    next_ = route[pos]
                    d_pj  = np.linalg.norm(all_xy[prev] - all_xy[j])
                    d_jn  = np.linalg.norm(all_xy[j]    - all_xy[next_])
                    d_pn  = np.linalg.norm(all_xy[prev] - all_xy[next_])
                    detour = d_pj + d_jn - d_pn
                    if detour < best_cost:
                        best_cost = detour
                        best_pos  = pos
                        best_route = r_idx
            if best_cost <= self.threshold and best_pos is not None:
                routes[best_route].insert(best_pos, j)

        # Tính total cost
        total_dist = 0.0
        n_served = sum(len(r) - 2 for r in routes)  # trừ 2 depot nodes
        for route in routes:
            for k in range(len(route) - 1):
                total_dist += np.linalg.norm(all_xy[route[k]] - all_xy[route[k+1]])

        return {
            'total_cost': total_dist,
            'qos': n_served / n,
            'routes': routes,
        }

    def _nearest_neighbor_init(self, instance: Instance, customer_list: list) -> list:
        """Greedy nearest-neighbor init cho từng xe."""
        n  = instance.n_customers
        m  = instance.n_vehicles
        all_xy = np.vstack([instance.depot_xy, instance.cust_xy])
        
        unvisited = set(customer_list)
        routes = [[0, 0] for _ in range(m)]  # mỗi route bắt đầu và kết thúc ở depot
        caps   = [MAX_CAPACITY] * m

        # Round-robin assignment
        v_idx = 0
        while unvisited:
            route = routes[v_idx]
            cap   = caps[v_idx]
            cur   = route[-2]  # node hiện tại (trước depot cuối)
            
            # Nearest feasible customer
            best, best_dist = None, float('inf')
            for j in unvisited:
                if instance.cust_q[j-1] <= cap:
                    d = np.linalg.norm(all_xy[cur] - all_xy[j])
                    if d < best_dist:
                        best_dist = d
                        best = j
            
            if best is None:
                v_idx = (v_idx + 1) % m  # next vehicle
                if all(len(r) == 2 for r in routes):
                    break  # không thể phục vụ thêm
            else:
                route.insert(-1, best)  # insert trước depot cuối
                caps[v_idx] -= instance.cust_q[best-1]
                unvisited.remove(best)
        
        return routes