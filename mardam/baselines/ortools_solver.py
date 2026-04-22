# baselines/ortools_solver.py
"""
ORTools baseline cho DS-CVRPTW.

Cần cài đặt: pip install ortools

Ba variants (Section 6.3):
  - ORTools (o): dùng optimistic speed V_NOM (không biết có tắc đường)
  - ORTools (e): dùng expected speed E[V] = (1-p_slow)*V_NOM + p_slow*V_SLOW
  - ORTools (d): dynamic replanning mỗi khi có customer mới xuất hiện
"""
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import numpy as np
from ..env.state import *
from ..env.instance import Instance


def _build_ortools_model(instance: Instance, speed: float, time_limit_s: int = 5):
    """Xây dựng và giải ORTools model cho 1 instance với speed cố định.
    
    Args:
        instance:     bài toán
        speed:        tốc độ giả định (km/min)
        time_limit_s: time limit solver (giây)
    
    Returns:
        routes: list of lists — routes[i] = [0, j1, j2, ..., 0] cho xe i
        total_cost: tổng khoảng cách
    """
    n = instance.n_customers
    m = instance.n_vehicles
    
    # Tất cả nodes: 0 = depot, 1..n = customers
    # ORTools cần distance matrix (int)
    all_xy = np.vstack([instance.depot_xy, instance.cust_xy])  # (n+1, 2)
    
    # Distance matrix tính theo thời gian (phút) = dist/speed
    # ORTools muốn integer -> nhân 100 để giữ độ chính xác
    SCALE = 100
    n_nodes = n + 1
    time_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist = np.linalg.norm(all_xy[i] - all_xy[j])
            time_matrix[i][j] = int(dist / speed * SCALE)
    
    # Service times
    service_times = [0] + [int(instance.cust_d[j] * SCALE) for j in range(n)]
    
    # Time windows (scaled)
    time_windows = [(0, int(HORIZON * SCALE))]  # depot
    for j in range(n):
        e = int(instance.cust_e[j] * SCALE)
        l = int(instance.cust_l[j] * SCALE)
        time_windows.append((e, l))
    
    # Demands & capacities
    demands = [0] + [int(instance.cust_q[j] * 10) for j in range(n)]  # x10
    capacity = int(MAX_CAPACITY * 10)

    # ── ORTools setup ──────────────────────────────────────────────
    manager = pywrapcp.RoutingIndexManager(n_nodes, m, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Travel time callback
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node   = manager.IndexToNode(to_idx)
        return time_matrix[from_node][to_node] + service_times[from_node]

    transit_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Time windows dimension
    routing.AddDimension(
        transit_cb,
        int(HORIZON * SCALE),  # max wait time
        int(HORIZON * SCALE),  # max time per vehicle
        False,                 # don't force start at 0
        'Time'
    )
    time_dim = routing.GetDimensionOrDie('Time')
    for loc_idx, (lo, hi) in enumerate(time_windows):
        idx = manager.NodeToIndex(loc_idx)
        time_dim.CumulVar(idx).SetRange(lo, hi)

    # Capacity constraint
    def demand_callback(from_idx):
        return demands[manager.IndexToNode(from_idx)]
    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, [capacity] * m, True, 'Capacity'
    )

    # Solver params
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(search_params)
    
    if not solution:
        return None, float('inf')

    # Extract routes
    routes = []
    total_cost = 0
    for v in range(m):
        route = [0]
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node)
                j = node - 1
                dist = np.linalg.norm(
                    all_xy[route[-2] if len(route) > 1 else 0] - all_xy[node]
                )
                total_cost += dist
            idx = solution.Value(routing.NextVar(idx))
        route.append(0)
        routes.append(route)
    
    return routes, total_cost


class ORToolsBaseline:
    """Wrapper cho ORTools solver với 3 speed variants."""

    def __init__(self, variant: str = 'o', p_slow: float = 0.1):
        """
        Args:
            variant: 'o' = optimistic (V_NOM), 'e' = expected speed, 'd' = dynamic
            p_slow:  xác suất tắc đường (dùng cho variant 'e')
        """
        assert variant in ('o', 'e', 'd')
        self.variant = variant
        self.p_slow  = p_slow
        
        if variant == 'o':
            self.speed = V_NOM
        elif variant == 'e':
            self.speed = (1 - p_slow) * V_NOM + p_slow * V_SLOW

    def solve(self, instance: Instance) -> dict:
        """Giải 1 instance, trả về routes và metrics."""
        speed = self.speed if self.variant != 'd' else V_NOM
        routes, cost = _build_ortools_model(instance, speed)
        return {'routes': routes, 'cost': cost}

    def evaluate_batch(self, instances: list[Instance]) -> dict:
        """Evaluate trên nhiều instances, trả về mean ± std cost."""
        costs = []
        for inst in instances:
            result = self.solve(inst)
            costs.append(result['cost'])
        costs = np.array(costs)
        return {
            'mean': float(costs.mean()),
            'std':  float(costs.std()),
        }