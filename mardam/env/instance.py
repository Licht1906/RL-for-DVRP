import numpy as np
from .state import (Instance, AREA_SIZE, HORIZON, EARLY_CUTOFF, LATE_CUTOFF, MAX_CAPACITY, DEMAND_MIN, DEMAND_MAX, SERVICE_DUR_MIN, SERVICE_DUR_MAX, TW_WIDTH_MIN, TW_WIDTH_MAX)

def generate_instance(
    n_customers: int,
    n_vehicles: int,
    p_dyn: float = 0.3,        # tỉ lệ khách dynamic (xuất hiện sau khi xe đã xuất phát)
    p_early: float = 0.5,      # trong số dynamic, tỉ lệ xuất hiện trong [0, EARLY_CUTOFF]
    tw_ratio: float = 0.7,     # tỉ lệ khách có TW constraint (0 = không ai, 1 = tất cả)
    rng: np.random.Generator = None,
) -> Instance:
    if rng is None:
        rng = np.random.default_rng()

    n = n_customers
    m = n_vehicles

    #Sinh tọa độ depot + customers 
    all_xy = rng.uniform(0, AREA_SIZE, size = (n + 1,2)).astype(np.float32)
    depot_xy = all_xy[0]            # depot index 0
    cust_xy = all_xy[1:]     # customers index 1->n

    # Sinh demand
    cust_q = rng.uniform(DEMAND_MIN, DEMAND_MAX, size = n)
    cust_q = np.round(cust_q,1).astype(np.float32)
    #Scalde down if total demand > total capacity
    total_demand = cust_q.sum()
    total_capacity = m*MAX_CAPACITY
    if total_demand > total_capacity:
        cust_q = cust_q * (total_capacity / total_demand) * 0.95
        cust_q = np.round(cust_q,1).astype(np.float32)
    
    # Sinh appearance time (dynamic and static)
    n_dyn = int(round(n*p_dyn))     # number of dynamic customers
    cust_b = np.zeros(n, dtype = np.float32)    # 0 = static
    if n_dyn > 0:
        dyn_idx = rng.choice(n, size = n_dyn, replace = False)
        n_early = int(round(n_dyn * p_early))       # ty le khach dynamic xuat hien som
        early_idx = dyn_idx[:n_early]               # nhom khach dynamic xuat hien som
        late_idx = dyn_idx[n_early:]                # nhom khach dynamic xuat hien muon
        # early appearance time  0 -> EARLY_CUTOFF
        cust_b[early_idx] = rng.uniform(1.0, EARLY_CUTOFF, size = len(early_idx))
        # late appearance time: EARLY_CUTOFF -> LATE_CUTOFF
        cust_b[late_idx] = rng.uniform(EARLY_CUTOFF + 1.0, LATE_CUTOFF, size = len(late_idx))

    # Sinh time window
    cust_d = rng.uniform(SERVICE_DUR_MIN, SERVICE_DUR_MAX, size = n).astype(np.float32) # random service duraticon for customers

    cust_e = np.zeros(n, dtype = np.float32)
    cust_l = np.full(n, HORIZON, dtype = np.float32)

    n_tw = int(round(n * tw_ratio))
    if n_tw > 0:
        tw_idx = rng.choice(n, size = n_tw, replace = False)
        for j in tw_idx:
            # TW Width random in range [30:90] minutes
            tw_width = rng.uniform(TW_WIDTH_MIN, TW_WIDTH_MAX)
            # Earliest reachable time from depot
            dist_depot = np.linalg.norm(cust_xy[j] - depot_xy)
            earliest_arrive = cust_b[j] + dist_depot / 0.9
            e_j = earliest_arrive + rng.uniform(0, tw_width * 0.5)
            l_j = e_j + tw_width
            # ensure finish before HORIZON
            dist_to_depot = np.linalg.norm(cust_xy[j] -depot_xy)
            latest_start = HORIZON - cust_d[j]  - dist_to_depot / 0.9
            l_j = min(l_j, latest_start)
            e_j = min(e_j, l_j - 10.0)
            if e_j >= 0 and l_j > e_j:
                cust_e[j] = float(e_j)
                cust_l[j] = float(l_j)

    return Instance(
        n_customers = n,
        n_vehicles = m,
        depot_xy = depot_xy,
        cust_xy = cust_xy,
        cust_q = cust_q,
        cust_e = cust_e,
        cust_l = cust_l,
        cust_d = cust_d,
        cust_b = cust_b,    
    )

def generate_dataset(
        num_samples: int,
        n_customers: int,
        n_vehicles:int,
        seed: int = 42,
        **kwargs,
) -> list[Instance]:
    rng = np.random.default_rng(seed)
    return[
        generate_instance(n_customers, n_vehicles, rng = rng, **kwargs)
        for _ in range(num_samples)
    ]




