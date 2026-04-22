import numpy as np
from typing import Optional, Tuple, Dict
from .state import *
from .instance import Instance


# sMMDP environment for DSCVRPTW
class DSCVRPTWEnv:
    def __init__(self, instance: Instance, p_slow: float = 0.1): #p_slow : xac suat tac duong
        self.instance = instance
        self.p_slow = p_slow
        self.n = instance.n_customers
        self.m = instance.n_vehicles

        # States array
        self.shared_state: Optional[np.ndarray] = None
        self.vehicle_states: Optional[np.ndarray] = None
        self.turn: int = 0          # index xe đang được chọn để phục vụ

        #Tracking
        self.done: bool = False
        self.step_count: int = 0
        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    # RESET environment về trạng thái ban đầu
    def reset(self) -> Tuple[Dict, np.ndarray]:
        self.shared_state = self.instance.to_shared_state() #(n+1, 8)
        self.vehicle_state = self.instance.to_vehicle_state()  #(m,4)
        self.done = False
        self.step_count = 0

        # First agent active : agent have avail_time min
        self.turn = self._get_turn()

        state_dict = self._get_state_dict()
        mask = self._compute_mask(self.turn)
        return state_dict, mask   # return state_dict : {'shared_state': shared_state, 'vehicle_state': vehicle_state, 'turn': int} , mask : (n+1,) boolean array actions allowed for current turn
    
    def step(self, action: int) -> Tuple[Dict, np.ndarray, float, bool, dict]:
        """Thực hiện action cho xe đang trong lượt.
        
        Args:
            action: 0 = về depot, j∈[1,n] = phục vụ customer j
        
        Returns:
            next_state_dict: trạng thái sau hành động
            next_mask:       mask cho xe tiếp theo
            reward:          reward ngay lập tức (âm)
            done:            True nếu episode kết thúc
            info:            dict chứa thông tin debug
        """
        assert not self.done, "Episode đã kết thúc, gọi reset() trước"
        i = self.turn   # xe đang hành động

        mask = self._compute_mask(i)
        assert mask[action] == 1, f"Action {action} không hợp lệ cho xe {i}"

        # Lấy thông tin xe hiện tại
        xi, yi = self.vehicle_state[i, VEH_X], self.vehicle_state[i, VEH_Y]
        t_i    = self.vehicle_state[i, VEH_T]  # availability time

        # Tọa độ đích
        if action == 0:  # về depot
            xj, yj = self.instance.depot_xy
            qj, ej, lj, dj = 0, 0, HORIZON, 0
        else:            # phục vụ customer action (1-indexed)
            j = action - 1   # 0-indexed trong arrays
            xj = self.shared_state[action, IDX_X]
            yj = self.shared_state[action, IDX_Y]
            qj = self.shared_state[action, IDX_Q]
            ej = self.shared_state[action, IDX_E]
            lj = self.shared_state[action, IDX_L]
            dj = self.shared_state[action, IDX_D]
        # ── Transition function (Eq. 5.7) ──────────────────────────
        # Travel time = distance / speed (stochastic)
        dist_ij  = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        speed    = self._sample_speed()
        tau_ij   = dist_ij / speed   # phút

        # Arrival time (có thể phải chờ nếu customer chưa ready)
        arrive_t = t_i + tau_ij
        start_service = max(arrive_t, ej)  # phải đợi nếu arrive < e_j
        finish_t = start_service + dj

        # ── Reward (Eq. 5.8) ────────────────────────────────────────
        lateness = max(0.0, arrive_t - lj)   # trễ so với due time
        reward = -dist_ij - C_LATE * lateness

        # ── Update shared state ─────────────────────────────────────
        if action > 0:
            self.shared_state[action, IDX_XI] = 0.0  # đánh dấu đã phục vụ

        # ── Update vehicle state ────────────────────────────────────
        self.vehicle_state[i, VEH_X]   = xj
        self.vehicle_state[i, VEH_Y]   = yj
        if action > 0:
            self.vehicle_state[i, VEH_CAP] -= qj
        self.vehicle_state[i, VEH_T]   = finish_t

        # ── Reveal dynamic customers ────────────────────────────────
        # Khách hàng xuất hiện nếu b_j <= thời điểm xe vừa kết thúc
        self._reveal_dynamic_customers(finish_t)

        # ── Check episode done ──────────────────────────────────────
        # Done khi TẤT CẢ xe đang ở depot (action == 0 vừa thực hiện)
        # Xe ở depot khi vị trí == depot VÀ avail_time đã được set
        self.step_count += 1
        self.done = self._check_done()

        # ── Completion reward ────────────────────────────────────────
        if self.done:
            n_pending = int(self.shared_state[1:, IDX_XI].sum())
            reward += -C_PEND * n_pending  # Eq. 5.9

        # ── Next turn ───────────────────────────────────────────────
        if not self.done:
            self.turn = self._get_turn()

        next_state_dict = self._get_state_dict()
        next_mask = self._compute_mask(self.turn) if not self.done else np.zeros(self.n + 1)

        info = {
            'travel_dist':  dist_ij,
            'lateness':     lateness,
            'step':         self.step_count,
            'vehicle_idx':  i,
            'action':       action,
        }
        return next_state_dict, next_mask, float(reward), self.done, info

    # ────────────────────────────────────────────────────────────────
    #  HELPER METHODS
    # ────────────────────────────────────────────────────────────────
    def _sample_speed(self) -> float:
        """Bimodal speed distribution (Section 5.1.4, Figure 5.4).
        
        Mix của 2 Gaussians:
        - Nominal: N(V_NOM, V_NOM_STD²) với prob (1 - p_slow)
        - Slow:    N(V_SLOW, V_SLOW_STD²) với prob p_slow
        """
        if self._rng.random() < self.p_slow:
            speed = self._rng.normal(V_SLOW, V_SLOW_STD)
        else:
            speed = self._rng.normal(V_NOM, V_NOM_STD)
        return max(speed, 0.05)  # clamp để tránh chia 0

    def _get_turn(self) -> int:
        """Turn function σ(s̄): trả về xe có availability_time nhỏ nhất.
        
        Tham chiếu: Section 5.1.4 (Turn function)
        """
        return int(np.argmin(self.vehicle_state[:, VEH_T]))

    def _compute_mask(self, vehicle_idx: int) -> np.ndarray:
        """Compute action mask Ξ^i(s̄) cho xe vehicle_idx.
        
        Action j hợp lệ khi (Section 5.1.3):
          1. j == 0 (depot): luôn hợp lệ
          2. j > 0: customer j phải pending (xi_j == 1)
                    + đã xuất hiện (b_j <= t_i của xe)
                    + xe đủ sức chứa (kappa_i >= q_j)
        
        Returns:
            mask: (n+1,) binary array
        """
        mask = np.zeros(self.n + 1, dtype=np.float32)
        
        # Depot luôn hợp lệ (action 0)
        mask[0] = 1.0

        t_i     = self.vehicle_state[vehicle_idx, VEH_T]
        kappa_i = self.vehicle_state[vehicle_idx, VEH_CAP]

        for j in range(1, self.n + 1):
            xi_j = self.shared_state[j, IDX_XI]   # pending flag
            b_j  = self.shared_state[j, IDX_B]    # appearance time
            q_j  = self.shared_state[j, IDX_Q]    # demand

            if xi_j == 1.0 and b_j <= t_i and kappa_i >= q_j:
                mask[j] = 1.0

        # Nếu không có customer hợp lệ -> phải về depot (đã set mask[0]=1)
        return mask

    def _reveal_dynamic_customers(self, current_time: float):
        """Reveal customers có b_j <= current_time mà chưa visible.
        
        Dynamic customer: b_j > 0 (static: b_j == 0)
        Khi reveal: chỉ set pending flag xi_j = 1 (nếu chưa từng được set)
        """
        for j in range(1, self.n + 1):
            b_j  = self.instance.cust_b[j - 1]  # 0-indexed trong instance
            xi_j = self.shared_state[j, IDX_XI]
            # Nếu là dynamic customer (b_j > 0), chưa visible (xi_j == 0),
            # và đã đến thời gian xuất hiện:
            if b_j > 0 and xi_j == 0.0 and b_j <= current_time:
                self.shared_state[j, IDX_XI] = 1.0

    def _check_done(self) -> bool:
        """Episode kết thúc khi tất cả xe đã về depot.
        
        Xe về depot khi: VEH_X == depot_x AND VEH_Y == depot_y
        (sau khi thực hiện action 0)
        """
        dx, dy = self.instance.depot_xy
        at_depot = (
            (self.vehicle_state[:, VEH_X] == dx) &
            (self.vehicle_state[:, VEH_Y] == dy)
        )
        return bool(at_depot.all())

    def _get_state_dict(self) -> dict:
        """Trả về state dưới dạng dict để MARDAM sử dụng."""
        return {
            'shared':   self.shared_state.copy(),    # (n+1, 8)
            'vehicles': self.vehicle_state.copy(),   # (m, 4)
            'turn':     self.turn,                   # int
        }

    def get_qos(self) -> float:
        """Quality of Service = tỉ lệ khách hàng đã được phục vụ."""
        n_served = int((self.shared_state[1:, IDX_XI] == 0).sum())
        return n_served / self.n

    # ────────────────────────────────────────────────────────────────
    #  BATCH INTERFACE (dùng khi training với nhiều instances song song)
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def batch_state_to_tensor(
        state_dicts: list[dict],
        device: str = 'cpu',
    ):
        """Convert list of state_dicts thành batched tensors cho MARDAM.
        
        Args:
            state_dicts: list B state_dicts
        Returns:
            batch: {
                'shared':   Tensor (B, n+1, 8),
                'vehicles': Tensor (B, m, 4),
                'turns':    Tensor (B,) long
            }
        """
        import torch
        shared   = torch.tensor(
            np.stack([sd['shared']   for sd in state_dicts]), device=device
        )
        vehicles = torch.tensor(
            np.stack([sd['vehicles'] for sd in state_dicts]), device=device
        )
        turns    = torch.tensor(
            [sd['turn'] for sd in state_dicts], dtype=torch.long, device=device
        )
        return {'shared': shared, 'vehicles': vehicles, 'turns': turns}
        






        

    

    




        

