"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import glob
import heapq
import math
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib
import time
import torch
import globalvar
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from scipy.interpolate import interp1d
from DataLoader.dataload import opendata
from others.dynamic_programming_heuristic import calc_distance_heuristic
from others.ReedsSheppPath import reeds_shepp_path_planning as rs
from others.car import move, check_car_collision, MAX_STEER, WB, plot_car, BUBBLE_R
from utils.utils import path_smoothness, visualize_single_data
import multiprocessing
XY_GRID_RESOLUTION = 0.5  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.8  # [m] path interpolate resolution
N_STEER = 5  # number of steer command

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost
RESULT_DIR = './Astar_results6'
SUMMARY_FILE = './Astar_summary6.txt'
show_animation = False
from matplotlib.path import Path as MplPath
def obstacle_blowup_quadrilateral(obstacles, blowup_distance):
    blown_up_obstacles = []
    for obs in obstacles:
        poly = np.array(obs)
        # 1. 强制逆时针
        signed_area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], 1) - 
                                   poly[:, 1] * np.roll(poly[:, 0], 1))
        if signed_area < 0:
            poly = poly[::-1]

        # 2. 计算法向量并外推
        edges = np.roll(poly, -1, axis=0) - poly
        edge_lengths = np.linalg.norm(edges, axis=1, keepdims=True)
        edge_lengths[edge_lengths < 1e-6] = 1e-6 
        unit_edges = edges / edge_lengths
        normals = np.stack([unit_edges[:, 1], -unit_edges[:, 0]], axis=1)

        n_curr = normals
        n_prev = np.roll(normals, 1, axis=0)
        denom = n_prev[:, 0] * n_curr[:, 1] - n_prev[:, 1] * n_curr[:, 0]
        denom[np.abs(denom) < 1e-6] = 1e-6
        
        d = blowup_distance
        delta_x = d * (n_prev[:, 1] - n_curr[:, 1]) / denom
        delta_y = d * (n_curr[:, 0] - n_prev[:, 0]) / denom
        delta = np.stack([delta_x, delta_y], axis=1)
        
        new_poly = poly + delta
        blown_up_obstacles.append(new_poly)

    return blown_up_obstacles # (M, 4, 2)
def get_obstacle_points_from_polygons(polygons, resolution=0.5):
    ox, oy = [], []
    polygons = obstacle_blowup_quadrilateral(polygons, blowup_distance=0.3)
    for poly_points in polygons:
        path = MplPath(poly_points)
        
        poly_arr = np.array(poly_points)
        min_x, min_y = np.min(poly_arr, axis=0)
        max_x, max_y = np.max(poly_arr, axis=0)
        
        x_range = np.arange(min_x, max_x + resolution, resolution)
        y_range = np.arange(min_y, max_y + resolution, resolution)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        
        # radius=0 表示包含边界
        mask = path.contains_points(points, radius=0.001) 
        
        valid_points = points[mask]
        
        ox.extend(valid_points[:, 0])
        oy.extend(valid_points[:, 1])

        for i in range(len(poly_points)):
            p1 = poly_points[i]
            p2 = poly_points[(i + 1) % len(poly_points)]
            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            num_pts = int(dist / resolution)
            ox.extend(np.linspace(p1[0], p2[0], num_pts))
            oy.extend(np.linspace(p1[1], p2[1], num_pts))
            
    return ox, oy

class Node:

    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


class Path:

    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost


class Config:

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = globalvar.planning_scale_.xmin
        min_y_m = globalvar.planning_scale_.ymin
        max_x_m = globalvar.planning_scale_.xmax
        max_y_m = globalvar.planning_scale_.ymax

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)


def calc_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        for d in [1, -1]:
            yield [steer, d]


def get_neighbors(current, config, ox, oy, kd_tree):
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list, direction_list = [], [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)
        direction_list.append(direction == 1)

    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l

    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, direction_list,
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    if n1.x_index == n2.x_index \
            and n1.y_index == n2.y_index \
            and n1.yaw_index == n2.yaw_index:
        return True
    return False


def analytic_expansion(current, goal, ox, oy, kd_tree):
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analytic_expansion(current, goal,
                                        c, ox, oy, kd_tree):
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c)

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        return True, f_path

    return False, None


def calc_rs_path_cost(reed_shepp_path):
    cost = 0.0
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in range(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    config = Config(tox, toy, xy_resolution, yaw_resolution)

    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, BUBBLE_R)

    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),
                        calc_index(start_node, config)))
    final_path = None
    start_time = time.time()
    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return Path([], [], [], [], 0)
        now_time = time.time()
        if now_time - start_time > 30.0:
            print("Error: Cannot find path, Timeout")
            return Path([], [], [], [], 0)
        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)
        dist_to_goal = np.hypot(current.x_list[-1] - goal_node.x_list[-1], 
                        current.y_list[-1] - goal_node.y_list[-1])
        if dist_to_goal < 5.0 or (len(closedList) % 10 == 0):
            is_updated, final_path = update_node_with_analytic_expansion(
                current, goal_node, config, ox, oy, obstacle_kd_tree)

            if is_updated:
                break

        for neighbor in get_neighbors(current, config, ox, oy,
                                      obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor_index not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, final_path)
    return path


def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False


def calc_index(node, c):
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


def test_single(index):
    save_file = f'/home/qian/dataset_V7_labels_6/{index}.npz'
    data_file = f'/home/qian/dataset_V7/{index}.npz'
    save_path = os.path.join(RESULT_DIR, f'res_{index}.json')
    img_save_dir = './logs/HybridAstar'
    os.makedirs(img_save_dir, exist_ok=True)
    if os.path.exists(save_path):
        return
    data = opendata(data_file)
    obstacles = data['obstacles_vertices']
    target = data['target']
    
    ymax = globalvar.planning_scale_.ymax
    ymin = globalvar.planning_scale_.ymin
    xmin = globalvar.planning_scale_.xmin
    xmax = globalvar.planning_scale_.xmax
    bound = [[
        [xmax, ymax],
        [xmax, ymax + 5],
        [xmin, ymax + 5],
        [xmin, ymax]
    ],
    [
        [xmax, ymin],
        [xmax, ymin - 5],
        [xmin, ymin - 5],
        [xmin, ymin]
    ]]
    obstacles = np.vstack((obstacles, bound))
    results = []
    target_theta = -np.pi/2
    # Set Initial parameters
    ox, oy = get_obstacle_points_from_polygons(obstacles, resolution=0.2)
    min_len = float('inf')
    while target_theta <= math.pi/2:
        test_metrics = {'time': 0.0, 'length':0.0, 'smoothness':0.0, 'objective_distance':0.0, 'curvature':0.0, 'path': None}
        start = [0.0, 0.0, 0.0]
        goal = [target[0], target[1], target_theta]

        begin_time = time.time()
        path = hybrid_a_star_planning(
            start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        end_time = time.time()
        
        time_consumption = end_time - begin_time
        # print(f"Hybrid A* planning time: {time_consumption:.3f} sec for target_theta={np.rad2deg(target_theta):.1f} deg")
        if not path.x_list:
            print("No path found")
            target_theta += np.deg2rad(30.0)
            continue
        else:
            # 增加batch维度进行可视化
            path_x = path.x_list
            path_y = path.y_list
            path_xy = np.array([[path_x[i], path_y[i]] for i in range(len(path_x))])
            path_length = np.sum(np.hypot(np.diff(path_x), np.diff(path_y)))
            # print("Path length:", path_length)
            if path_length < min_len:
                min_len = path_length
                # visualize_single_data(data, path_xy, save_path=img_save_dir, i = index)
            target_theta += np.deg2rad(10.0)
            # continue
        path_x = path.x_list
        path_y = path.y_list
        path_yaw = path.yaw_list
        path_xy = np.array([[path_x[i], path_y[i]] for i in range(len(path_x))])
        path_all = np.array([[path_x[i], path_y[i], path_yaw[i]] for i in range(len(path_x))])
        path_length = np.sum(np.hypot(np.diff(path_x), np.diff(path_y)))
        smoothness, curvature_score = path_smoothness(path_xy)

        test_metrics['time'] = time_consumption
        test_metrics['length'] = path_length
        test_metrics['smoothness'] = smoothness
        test_metrics['objective_distance'] = np.hypot(path_x[-1] - goal[0], path_y[-1] - goal[1])
        test_metrics['curvature'] = curvature_score
        test_metrics['path'] = path_all
        results.append(test_metrics)
        
        target_theta += 0.5
    if not results:
        print("No valid paths found.")
        return
    best_result = results[0]
    for res in results:
        if res['length'] < best_result['length']:
            best_result = res
    best_path = best_result['path']
    # np.savez(save_file, path=best_path)
    # print(f"Saved label to {save_file}")
    print('index=', index, 'time=', best_result['time'], 'length=', best_result['length'],
          'smoothness=', best_result['smoothness'], 'objective_distance=', best_result['objective_distance'],
          'curvature=', best_result['curvature'])
    
    # del best_result['path']
    # with open(save_path, 'w') as f:
    #     json.dump(best_result, f)
    
    return

def normalize_path_points(raw_path, num_points=40):
    raw_path = np.array(raw_path)
    
    if raw_path is None or len(raw_path) < 2:
        # 直接填充起点，保持形状一致
        if len(raw_path) == 1:
            return np.tile(raw_path[0, :2], (num_points, 1))
        else:
            return np.zeros((num_points, 2))

    path_xy = raw_path[:, :2]
    
    # diffs[i] = point[i+1] - point[i]
    diffs = np.diff(path_xy, axis=0)
    
    # norms[i] = len(diffs[i])
    distances = np.linalg.norm(diffs, axis=1)
    
    cum_dist = np.insert(np.cumsum(distances), 0, 0)
    
    total_length = cum_dist[-1]
    
    if total_length == 0:
        return np.tile(path_xy[0], (num_points, 1))

    fx = interp1d(cum_dist, path_xy[:, 0], kind='linear')
    fy = interp1d(cum_dist, path_xy[:, 1], kind='linear')
    
    target_dists = np.linspace(0, total_length, num_points)
    
    new_x = fx(target_dists)
    new_y = fy(target_dists)
    
    normalized_path = np.column_stack((new_x, new_y))
    
    return normalized_path

def generate_label(index):
    save_file = f'/home/qian/dataset_V7_labels_6/{index}.npz'
    if os.path.exists(save_file):
        print(f"Label for index {index} already exists.")
        return

    data_file = f'/home/qian/dataset_V7/{index}.npz'
    data = opendata(data_file)
    obstacles = data['obstacles_vertices']
    target = data['target']
    ymax = globalvar.planning_scale_.ymax
    ymin = globalvar.planning_scale_.ymin
    xmin = globalvar.planning_scale_.xmin
    xmax = globalvar.planning_scale_.xmax
    bound = [[
        [xmax, ymax],
        [xmax, ymax + 5],
        [xmin, ymax + 5],
        [xmin, ymax]
    ],
    [
        [xmax, ymin],
        [xmax, ymin - 5],
        [xmin, ymin - 5],
        [xmin, ymin]
    ]]
    obstacles = np.vstack((obstacles, bound))
    results = []
    delta_theta = np.deg2rad(30.0)
    target_thetas = [0, delta_theta, -delta_theta, 2*delta_theta, -2*delta_theta, 3*delta_theta, -3*delta_theta]
    target_thetas = np.array(target_thetas)
    # Set Initial parameters
    ox, oy = get_obstacle_points_from_polygons(obstacles, resolution=0.2)
    for target_theta in target_thetas:
        test_metrics = {'time': 0.0, 'length':0.0, 'smoothness':0.0, 'objective_distance':0.0, 'curvature':0.0}
        start = [0.0, 0.0, 0.0]
        goal = [target[0], target[1], target_theta]
        begin_time = time.time()
        path = hybrid_a_star_planning(
            start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        end_time = time.time()
        time_consumption = end_time - begin_time
        if not path.x_list:
            print("No path found")
            target_theta += np.deg2rad(30.0)
            continue
        else:
            path_x = path.x_list
            path_y = path.y_list
            path_xy = np.array([[path_x[i], path_y[i]] for i in range(len(path_x))])
            path_xy = normalize_path_points(path_xy, num_points=40)
            np.savez(save_file, path=path_xy)
            print(f"Saved label for index {index} with target_theta={np.rad2deg(target_theta):.1f} deg")
            return
    return

def analyze_results():
    """
    读取所有保存的结果文件并计算统计数据
    """
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

            if 'length' in data and data['length'] > 0:
                stats['success_count'] += 1
                stats['time_consumption'].append(data['time'])
                stats['path_length'].append(data['length'])
                stats['path_smoothness'].append(data['smoothness'])
                stats['curvature'].append(data['curvature'])
            else:
                stats['time_consumption'].append(data['time'])
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 计算平均值
    success_rate = stats['success_count'] / total_cases if total_cases > 0 else 0
    avg_time = np.mean(stats['time_consumption']) if stats['time_consumption'] else 0
    avg_len = np.mean(stats['path_length']) if stats['path_length'] else 0
    avg_smooth = np.mean(stats['path_smoothness']) if stats['path_smoothness'] else 0
    avg_curv = np.mean(stats['curvature']) if stats['curvature'] else 0

    # 生成报告文本
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

if __name__ == '__main__':
    save_dir = '/home/qian/dataset_V7_labels_6'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    # seed = 42
    # np.random.seed(seed)
    # random.seed(seed)
    # index = 28085
    # generate_label(index)

    # data_file = '/home/qian/dataset_V7_labels/28085.npz'
    # data = opendata(data_file)
    # print(data['path'])
    # plt.plot(data['path'][:,0], data['path'][:,1], '-o')
    # plt.axis('equal')
    # plt.show()
    
    # index = range(0,1000)
    # cpu_num = multiprocessing.cpu_count()
    # with multiprocessing.Pool(processes=25) as pool:
    #     pool.map(generate_label, index)
    test_single(74)
    # index = range(0,100)
    # cpu_num = multiprocessing.cpu_count()
    # with multiprocessing.Pool(processes=15) as pool:
    #     pool.map(test_single, index)

    # analyze_results()