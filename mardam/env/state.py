from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

#Khai bao hang so
AREA_SIZE = 100.0                #khu vuc 100x100 pham vi bai toan
HORIZON = 480.0                  #tong thoi gian lam viec 8 tieng
EARLY_CUTOFF = 160.0             #moc thoi gian som 2 tieng 40p
LATE_CUTOFF = 320.0              #moc thoi gian muon 5 tieng 20p
MAX_CAPACITY = 20.0              #suc chua toi da cua xe 20M^3
DEMAND_MIN = 0.5                 #luong hang phuc vu toi thieu 0.5 
DEMAND_MAX = 4.0                 #luong hang phuc vu toi da 4.0
SERVICE_DUR_MIN = 5.0          # min
SERVICE_DUR_MAX = 30.0         # min
TW_WIDTH_MIN    = 30.0         # min
TW_WIDTH_MAX    = 90.0         # min

V_NOM = 0.9                    # van toc binh thuong 0.9 km / min
V_NOM_STD = 0.05                # do lech chuan van toc 0.05 km / min
V_SLOW = 0.3                   # van toc cham 0.3 km / min
V_SLOW_STD = 0.08               # do lech chuan van toc cham 0.08 km / min

C_LATE = 0.5                   # chi phi tre hen 
C_PEND = 5.0                   # chi phi phat voi moi hanh khach k dc phuc vu 

# shared state: (n+1) × 8 matrix, mỗi hàng = 1 customer
IDX_X    = 0   # tọa độ x
IDX_Y    = 1   # tọa độ y
IDX_Q    = 2   # demand (m³)
IDX_E    = 3   # time window start (ready time)
IDX_L    = 4   # time window end  (due time)
IDX_D    = 5   # service duration
IDX_B    = 6   # appearance time (0 nếu static customer)
IDX_XI   = 7   # pending flag: 1=chưa phục vụ, 0=đã phục vụ

# Vehicle state indices  s̄^i = [x, y, capacity, avail_time]
VEH_X    = 0   # vị trí x hiện tại (của customer vừa phục vụ)
VEH_Y    = 1   # vị trí y hiện tại
VEH_CAP  = 2   # sức chứa còn lại
VEH_T    = 3   # thời điểm xe sẵn sàng nhận nhiệm vụ tiếp

@dataclass
class Instance:
    n_customers: int                # so luong khach hang 
    n_vehicles: int                 # so luong xe
    depot_xy: np.ndarray            # toa do x,y cua depot (1,2)   
    # Customers feature
    cust_xy: np.ndarray
    cust_q: np.ndarray              # demands
    cust_e: np.ndarray              # time window start (ready time)
    cust_l: np.ndarray              # time window end  (due time)
    cust_d: np.ndarray              # service duration
    cust_b: np.ndarray              # appearance time (0 nếu static customer)

    #build shared state matrix shape(n+1,8)
    def to_shared_state(self) -> np.ndarray:
        n = self.n_customers
        s0 = np.zeros((n+1, 8), dtype = np.float32)
        #Depot index 0
        s0[0, IDX_X] = self.depot_xy[0]
        s0[0, IDX_Y] = self.depot_xy[1]
        s0[0, IDX_E] = 0.0
        s0[0, IDX_L] = HORIZON
        s0[0, IDX_XI] = 1 # depot always available
        #Customers index 1->n
        s0[1:, IDX_X] = self.cust_xy[:, 0]
        s0[1:, IDX_Y] = self.cust_xy[:, 1]
        s0[1:, IDX_Q] = self.cust_q
        s0[1:, IDX_E] = self.cust_e
        s0[1:, IDX_L] = self.cust_l
        s0[1:, IDX_D] = self.cust_d
        s0[1:, IDX_B] = self.cust_b
        #First, only static customers b==0 is pending
        s0[1:, IDX_XI] = (self.cust_b == 0.0).astype(np.float32)
        return s0
    
    #vehicle state shape(m,4)
    #all vehicles start at depot with full capacity and available at time 0
    def to_vehicle_state(self) -> np.ndarray:
        m = self.n_vehicles
        sv = np.zeros((m, 4), dtype = np.float32)
        sv[:, VEH_X] = self.depot_xy[0]
        sv[:, VEH_Y] = self.depot_xy[1]
        sv[:, VEH_CAP] = MAX_CAPACITY
        sv[:, VEH_T] = 0.0
        return sv
    

