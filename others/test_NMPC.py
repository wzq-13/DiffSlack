import os
import random
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from DataLoader.dataload import opendata
import globalvar
from utils.utils import path_smoothness, visualize_single_data
import multiprocessing
import glob
from tqdm import tqdm 
import json

@dataclass
class VehicleConfig:
    length: float = globalvar.vehicle_geometrics_.vehicle_length
    width: float = globalvar.vehicle_geometrics_.vehicle_width  
    wheelbase: float = globalvar.vehicle_geometrics_.vehicle_wheelbase  
    rear_overhang: float = 1.0

    max_steer: float = 0.7 
    max_vel: float = 2.4 
    min_vel: float = 0.0 
    max_acc: float = 20.0 

    # 避障安全参数
    safe_margin: float = 0.0 

@dataclass
class NMPCConfig:
    T: int = 40         
    dt: float = 0.5      
    w_goal: float = 5.0  
    w_smooth: float = 1.0 
    w_input: float = 1.0  
    alpha: float = 10.0   


class NMPCPlanner:
    def __init__(self, vehicle_cfg, mpc_cfg):
        self.v_cfg = vehicle_cfg
        self.mpc_cfg = mpc_cfg
        
        segment_len = self.v_cfg.length / 3.0
        
        half_seg = segment_len / 2.0
        half_width = self.v_cfg.width / 2.0
        self.car_radius = np.sqrt(half_seg**2 + half_width**2)
        
        geom_center_to_rear_axle = (self.v_cfg.length / 2.0) - self.v_cfg.rear_overhang
        
        offset_from_geom_center = np.array([-self.v_cfg.length/3.0, 0.0, self.v_cfg.length/3.0])
        
        self.circle_offsets = offset_from_geom_center + geom_center_to_rear_axle

    def get_half_planes(self, vertices):
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += (vertices[i, 0] * vertices[j, 1])
            area -= (vertices[j, 0] * vertices[i, 1])
        if area < 0: vertices = vertices[::-1]
            
        planes = []
        for i in range(4):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 4]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            A, B = normal[0], normal[1]
            C = -(A * p1[0] + B * p1[1])
            planes.append((A, B, C))
        return planes

    def plan(self, start_pose, target_pos, obs_polygons):
        opti = ca.Opti()
        T, dt = self.mpc_cfg.T, self.mpc_cfg.dt
        L_wheel = self.v_cfg.wheelbase
        
        X = opti.variable(3, T + 1)
        x, y, theta = X[0, :], X[1, :], X[2, :]
        U = opti.variable(2, T)
        v, delta = U[0, :], U[1, :]
        
        obj = self.mpc_cfg.w_goal * ((x[-1] - target_pos[0])**2 + (y[-1] - target_pos[1])**2)
        for k in range(T):
            obj += self.mpc_cfg.w_input * (delta[k]**2 + 0.1 * v[k]**2)
            if k < T - 1:
                obj += self.mpc_cfg.w_smooth * (delta[k+1] - delta[k])**2
                obj += 10 * (v[k+1] - v[k])**2
        opti.minimize(obj)
        
        opti.subject_to(X[:, 0] == start_pose)
        for k in range(T):
            beta = ca.atan(ca.tan(delta[k]) / 2.0)
            x_next = x[k] + v[k] * ca.cos(theta[k]) * dt
            y_next = y[k] + v[k] * ca.sin(theta[k]) * dt
            theta_next = theta[k] + (v[k] / L_wheel) * ca.tan(delta[k]) * dt
            opti.subject_to(X[:, k+1] == ca.vertcat(x_next, y_next, theta_next))
            
        opti.subject_to(opti.bounded(-self.v_cfg.max_steer, delta, self.v_cfg.max_steer))
        opti.subject_to(opti.bounded(self.v_cfg.min_vel, v, self.v_cfg.max_vel))
        
        obs_eqs = [self.get_half_planes(obs) for obs in obs_polygons]
        total_radius = self.car_radius + self.v_cfg.safe_margin
        alpha = self.mpc_cfg.alpha
        
        for k in range(1, T + 1): 
            c, s = ca.cos(theta[k]), ca.sin(theta[k])
            for offset in self.circle_offsets:
                cx = x[k] + offset * c
                cy = y[k] + offset * s
                for planes in obs_eqs:
                    dists = []
                    for (A, B, C) in planes:
                        dists.append(A * cx + B * cy + C)
                    
                    # LSE Trick
                    d_vec = ca.vertcat(*dists)
                    h_vec = -alpha * d_vec 
                    h_max = ca.mmax(h_vec)
                    lse = h_max + ca.log(ca.sum1(ca.exp(h_vec - h_max)))
                    dist_approx = - (1/alpha) * lse
                    
                    opti.subject_to(dist_approx + total_radius <= 0)

        opti.set_initial(x, np.linspace(start_pose[0], target_pos[0], T + 1))
        opti.set_initial(y, np.linspace(start_pose[1], target_pos[1], T + 1))
        opti.set_initial(v, 2.0)
        
        opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_cpu_time": 300}
        opti.solver("ipopt", opts)
        
        try:
            sol = opti.solve()
            return True, {
                "x": sol.value(x), "y": sol.value(y), "theta": sol.value(theta),
                "v": sol.value(v), "delta": sol.value(delta)
            }
        except Exception as e:
            print(f"IPOPT failed: {e}")
            return False, {}

RESULT_DIR = '.NMPC/NMPC'
SUMMARY_FILE = './NMPC_summary_report.txt'

def test_single(index):
    img_save_dir = './logs/NMPC'
    os.makedirs(img_save_dir, exist_ok=True)
    
    test_metrics = {'time': 0.0, 'length':0.0, 'smoothness':0.0, 'curvature':0.0, 'success': False}
    my_car = VehicleConfig()
    mpc_params = NMPCConfig(T=40, dt=0.5)
    planner = NMPCPlanner(my_car, mpc_params)
    data_file = f'/home/qian/dataset_V7/{index}.npz'
    data = opendata(data_file)
    obstacles = data['obstacles_vertices']
    target = data['target']
    start = [0, 0, 0]
    start_time = time.time()
    success, res = planner.plan(start, target, obstacles)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Test {index} elapsed time: {elapsed_time:.3f} s")
    test_metrics['time'] = elapsed_time
    # print(success)
    if success:
        test_metrics['length'] = np.sum(np.sqrt(np.diff(res['x'])**2 + np.diff(res['y'])**2))
        path_numpy = np.vstack((res['x'], res['y'], res['theta'])).T
        visualize_single_data(data, path_numpy, save_path=img_save_dir, i = index)
        smoothness, curvature_score = path_smoothness(path_numpy)
        test_metrics['smoothness'] = smoothness
        test_metrics['curvature'] = curvature_score
        test_metrics['success'] = True
    else:
        test_metrics['success'] = False
    
    # save_path = os.path.join(RESULT_DIR, f'res_{index}.json')
    # with open(save_path, 'w') as f:
    #     json.dump(test_metrics, f)

    # return None

def generate_labels_for_IL(index):
    save_dir = f'/home/qian/dataset_V7_NMPC-label/{index}.npz'
    if os.path.exists(save_dir):
        return
    my_car = VehicleConfig()
    mpc_params = NMPCConfig(T=40, dt=0.5)
    planner = NMPCPlanner(my_car, mpc_params)
    data_file = f'/home/qian/dataset_V7/{index}.npz'
    data = opendata(data_file)
    obstacles = data['obstacles_vertices']
    target = data['target']
    start = [0, 0, 0]
    success, res = planner.plan(start, target, obstacles)
    if success:
        path_numpy = np.vstack((res['x'][1:], res['y'][1:])).T
        new_data = {
            "path": path_numpy,
            "obstacles_vertices": obstacles,
            "target": target
        }
        np.savez(save_dir, **new_data)
        # print(f"Generated label for index {index}")

def analyze_results():
    print("Start analyzing results...")
    json_files = glob.glob(os.path.join(RESULT_DIR, 'res_*.json'))
    
    total_cases = len(json_files)
    if total_cases == 0:
        print("No result files found!")
        return

    stats = {
        'time_consumption': [],
        'path_length': [],
        'path_smoothness': [],
        'curvature': [],
        'success_count': 0
    }

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if data['success']:
                stats['success_count'] += 1
                stats['time_consumption'].append(data['time'])
                stats['path_length'].append(data['length'])
                stats['path_smoothness'].append(data['smoothness'])
                stats['curvature'].append(data['curvature'])
            else:
                stats['time_consumption'].append(data['time'])
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    success_rate = stats['success_count'] / total_cases if total_cases > 0 else 0
    avg_time = np.mean(stats['time_consumption']) if stats['time_consumption'] else 0
    avg_len = np.mean(stats['path_length']) if stats['path_length'] else 0
    avg_smooth = np.mean(stats['path_smoothness']) if stats['path_smoothness'] else 0
    avg_curv = np.mean(stats['curvature']) if stats['curvature'] else 0

    report = (
        "================ TEST REPORT ================\n"
        f"Total Cases Processed: {total_cases}\n"
        f"Success Rate: {success_rate * 100:.2f}%\n"
        f"Total Success: {stats['success_count']}\n"
        "---------------------------------------------\n"
        f"Avg Time Consumption: {avg_time:.4f} s\n"
        f"Avg Path Length:      {avg_len:.4f}\n"
        f"Avg Smoothness:       {avg_smooth:.4f}\n"
        f"Avg Curvature:        {avg_curv:.4f}\n"
        "=============================================\n"
    )

    print(report)
    
    with open(SUMMARY_FILE, 'w') as f:
        f.write(report)
        # json.dump(stats, f) 
    
    print(f"Summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    now_time = time.strftime("%Y%m%d-%H%M%S")
    # now_time = '20260508-164758'  #NMPC/NMPC_20260508-164758
    RESULT_DIR = f'NMPC/NMPC_{now_time}'
    SUMMARY_FILE = f'NMPC/NMPC_{now_time}/NMPC_summary.txt'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # indices = range(0, 100)
    # with multiprocessing.Pool(processes=10) as pool:
    #     results_iterator = pool.imap_unordered(test_single, indices)
        
    #     for _ in tqdm(results_iterator, total=len(indices)):
    #         pass
    test_single(74)
    # # print("All tasks completed.")
    # analyze_results()
    # index = random.randint(0, 20000)
    # test_single(index)
    # generate_labels_for_IL(index)
    
    # index = range(0, 2000)
    # for i in index:
    #     generate_labels_for_IL(i)