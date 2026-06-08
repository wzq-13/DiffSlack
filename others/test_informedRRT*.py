"""
Informed RRT* path planning with Polygon Obstacles

Modified to support irregular quadrilateral obstacles.
"""
import json
import os
import sys
import pathlib
import copy
import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import globalvar
from DataLoader.dataload import opendata
from utils.utils import path_smoothness, visualize_single_data
import multiprocessing
import glob
from tqdm import tqdm
import torch
try:
    from utils.angle import rot_mat_2d
except ImportError:
    def rot_mat_2d(angle):
        return np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])

show_animation = False

def obstacle_blowup_quadrilateral(obstacles, blowup_distance):
    """
    对四边形障碍物进行膨胀，并强制保持四边形形状。
    """
    blown_up_obstacles = []
    for obs in obstacles:
        poly = np.array(obs)
        signed_area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], 1) - 
                                   poly[:, 1] * np.roll(poly[:, 0], 1))
        if signed_area < 0:
            poly = poly[::-1]

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

    return blown_up_obstacles
class InformedRRTStar:

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=0.5,
                 goal_sample_rate=10, max_iter=5000):

        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.x_min, self.x_max = rand_area[0], rand_area[1]
        self.y_min, self.y_max = rand_area[2], rand_area[3]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = None

    def informed_rrt_star_search(self, animation=True):
        self.node_list = [self.start]
        c_best = float('inf')
        solution_set = set()
        path = None

        c_min = math.hypot(self.start.x - self.goal.x,
                           self.start.y - self.goal.y)
        x_center = np.array([[(self.start.x + self.goal.x) / 2.0],
                             [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / c_min],
                       [(self.goal.y - self.start.y) / c_min], [0]])

        e_theta = math.atan2(a1[1, 0], a1[0, 0])
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        m = a1 @ id1_t
        u, s, vh = np.linalg.svd(m, True, True)
        c = u @ np.diag(
            [1.0, 1.0,
             np.linalg.det(u) * np.linalg.det(np.transpose(vh))]) @ vh

        for i in range(self.max_iter):
            rnd = self.informed_sample(c_best, c_min, x_center, c)
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[n_ind]
            
            theta = math.atan2(rnd[1] - nearest_node.y,
                               rnd[0] - nearest_node.x)
            new_node = self.get_new_node(theta, n_ind, nearest_node)
            
            no_collision = self.check_line_collision(nearest_node, new_node)

            if no_collision:
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)

                self.node_list.append(new_node)
                self.rewire(new_node, near_inds)

                if self.is_near_goal(new_node):
                    if self.check_line_collision(new_node, self.goal):
                        solution_set.add(new_node)
                        last_index = len(self.node_list) - 1
                        temp_path = self.get_final_course(last_index)
                        temp_path_len = self.get_path_len(temp_path)
                        if temp_path_len < c_best:
                            path = temp_path
                            c_best = temp_path_len
            if animation and i % 5 == 0:
                self.draw_graph(x_center=x_center, c_best=c_best, c_min=c_min,
                                e_theta=e_theta, rnd=rnd)

        return path

    def choose_parent(self, new_node, near_inds):
        if len(near_inds) == 0:
            return new_node

        d_list = []
        for i in near_inds:
            dx = new_node.x - self.node_list[i].x
            dy = new_node.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            if self.check_line_collision(self.node_list[i], new_node):
                d_list.append(self.node_list[i].cost + d)
            else:
                d_list.append(float('inf'))

        min_cost = min(d_list)
        min_ind = near_inds[d_list.index(min_cost)]

        if min_cost == float('inf'):
            return new_node

        new_node.cost = min_cost
        new_node.parent = min_ind

        return new_node

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt(math.log(n_node) / n_node)
        d_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for
                  node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    def informed_sample(self, c_max, c_min, x_center, c):
        if c_max < float('inf'):
            r = [c_max / 2.0, math.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 math.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            rl = np.diag(r)
            x_ball = self.sample_unit_ball()
            rnd = np.dot(np.dot(c, rl), x_ball) + x_center
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            rnd = self.sample_free_space()
        return rnd

    @staticmethod
    def sample_unit_ball():
        a = random.random()
        b = random.random()
        if b < a:
            a, b = b, a
        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])

    def sample_free_space(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.x_min, self.x_max),
                    random.uniform(self.y_min, self.y_max)]
        else:
            rnd = [self.goal.x, self.goal.y]
        return rnd

    @staticmethod
    def get_path_len(path):
        path_len = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            path_len += math.hypot(node1_x - node2_x, node1_y - node2_y)
        return path_len

    @staticmethod
    def line_cost(node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in
                  nodes]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_new_node(self, theta, n_ind, nearest_node):
        new_node = copy.deepcopy(nearest_node)
        new_node.x += self.expand_dis * math.cos(theta)
        new_node.y += self.expand_dis * math.sin(theta)
        new_node.cost += self.expand_dis
        new_node.parent = n_ind
        return new_node

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    def rewire(self, new_node, near_inds):
        n_node = len(self.node_list)
        for i in near_inds:
            near_node = self.node_list[i]
            d = math.hypot(near_node.x - new_node.x, near_node.y - new_node.y)
            s_cost = new_node.cost + d

            if near_node.cost > s_cost:
                if self.check_line_collision(near_node, new_node):
                    near_node.parent = n_node - 1
                    near_node.cost = s_cost


    def check_line_collision(self, node1, node2):
        """
        检查 node1 到 node2 的连线是否与任何多边形障碍物碰撞
        """
        p1 = np.array([node1.x, node1.y])
        p2 = np.array([node2.x, node2.y])

        for obstacle in self.obstacle_list:
            if self.is_inside_polygon(p2, obstacle):
                return False # Collision
            
            n_vertices = len(obstacle)
            for i in range(n_vertices):
                v1 = np.array(obstacle[i])
                v2 = np.array(obstacle[(i + 1) % n_vertices])
                
                if self.is_intersect(p1, p2, v1, v2):
                    return False # Collision
                    
        return True # No Collision

    def check_collision(self, near_node, theta, d):
        tmp_node = copy.deepcopy(near_node)
        end_x = tmp_node.x + math.cos(theta) * d
        end_y = tmp_node.y + math.sin(theta) * d
        end_node = Node(end_x, end_y)
        return self.check_line_collision(tmp_node, end_node)

    @staticmethod
    def is_inside_polygon(point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def is_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    # ========================================================

    def get_final_course(self, last_index):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def draw_graph(self, x_center=None, c_best=None, c_min=None, e_theta=None,
                   rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            if c_best != float('inf'):
                self.plot_ellipse(x_center, c_best, c_min, e_theta)

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x],
                             [node.y, self.node_list[node.parent].y], "-g")
        
        ax = plt.gca()
        for ob in self.obstacle_list:
            poly = patches.Polygon(ob, closed=True, facecolor='black')
            ax.add_patch(poly)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([self.x_min, self.x_max, self.y_min, self.y_max])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_ellipse(x_center, c_best, c_min, e_theta):
        a = math.sqrt(c_best ** 2 - c_min ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - e_theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        fx = rot_mat_2d(-angle) @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xc")
        plt.plot(px, py, "--c")


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

RESULT_DIR = './RRT_experiment_results'
SUMMARY_FILE = './final_summary_report.txt'
SAVE_DIR = '/mnt/sim/carla/carla-ue4-0.9.16/PythonAPI/examples/path_data/RRT'
def linear_interpolation(path, resolution=0.1):
    """
    path: [[x1, y1], [x2, y2], ...]
    """
    path = np.array(path)
    new_path = []
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i+1]
        dist = np.linalg.norm(end_point - start_point)
        
        num_points = int(dist / resolution)+1
        if num_points < 1:
            num_points = 1
            
        for j in range(num_points):
            alpha = j / num_points
            interpolated_point = start_point * (1 - alpha) + end_point * alpha
            new_path.append(interpolated_point)
            
    new_path.append(path[-1]) 
    return np.array(new_path)

def test_single(index):
    test_metrics = {'index': index, 'time': 0.0, 'length':0.0, 'smoothness':0.0, 'curvature':0.0, 'success': False}
    data_file = f'/home/qian/dataset_V7/{index}.npz'
    data = opendata(data_file)
    obstacles = data['obstacles_vertices']
    target = data['target']
    img_save_dir = './logs/RRT'
    os.makedirs(img_save_dir, exist_ok=True)
    
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
    obstacle_list = np.vstack((obstacles, bound))
    obstacle_list = obstacle_blowup_quadrilateral(obstacle_list, blowup_distance=1.2)

    # Set params
    rrt = InformedRRTStar(start=[0, 0], goal=[target[0], target[1]], rand_area=[xmin, xmax, ymin, ymax],
                          obstacle_list=obstacle_list)
    begin_time = time.time()
    path = rrt.informed_rrt_star_search(animation=show_animation)
    end_time = time.time()
    time_consumption = end_time - begin_time
    if path is not None:
        path = linear_interpolation(path, resolution=0.8)
        path_length = rrt.get_path_len(path)
        path_numpy = np.array(path)
        visualize_single_data(data, path_numpy, save_path=img_save_dir, i = index)
        # data_b = {}
        # data_b['obstacles_vertices'] = torch.tensor(data['obstacles_vertices'], dtype=torch.float64).unsqueeze(0)
        # data_b['target'] = torch.tensor(data['target'], dtype=torch.float64).unsqueeze(0)
        print(f'path shape: {path_numpy.shape}')
        smoothness, curvature_score = path_smoothness(path_numpy)
        # path_torch_b = torch.tensor(path_numpy, dtype=torch.float64).unsqueeze(0)
        # visualize_data_batch_paper(data_b, path_torch_b, save_path=f'imgs/InformedRRTstar{index}')
        test_metrics['time'] = time_consumption
        test_metrics['length'] = path_length
        test_metrics['smoothness'] = smoothness
        test_metrics['curvature'] = curvature_score
        test_metrics['success'] = True
        print('index=', index, 'time=', time_consumption, 'length=', path_length,
              'smoothness=', smoothness, 'curvature=', curvature_score)
    else:
        print("No valid path found.")
        test_metrics['time'] = time_consumption
        test_metrics['length'] = 0.0
        test_metrics['smoothness'] = 0.0
        test_metrics['curvature'] = 0.0
        test_metrics['success'] = False

    # save_path = os.path.join(RESULT_DIR, f'res_{index}.json')
    # with open(save_path, 'w') as f:
    #     json.dump(test_metrics, f)

    return None

def save_path_data(index):
    save_dir = SAVE_DIR
    data_file = f'/home/qian/dataset_V7/{index}.npz'
    data = opendata(data_file)
    obstacles = data['obstacles_vertices']
    target = data['target']
    save_file_name = f'batch_{index}.npy'
    save_path = os.path.join(save_dir, save_file_name)
    if os.path.exists(save_path):
        return
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
    obstacle_list = np.vstack((obstacles, bound))
    obstacle_list = obstacle_blowup_quadrilateral(obstacle_list, blowup_distance=1.2)

    # Set params
    rrt = InformedRRTStar(start=[0, 0], goal=[target[0], target[1]], rand_area=[xmin, xmax, ymin, ymax],
                          obstacle_list=obstacle_list)
    begin_time = time.time()
    path = rrt.informed_rrt_star_search(animation=show_animation)
    end_time = time.time()
    time_consumption = end_time - begin_time
    if path is not None:
        path = linear_interpolation(path, resolution=1.0)
        path_length = rrt.get_path_len(path)
        path_numpy = np.array(path)
        # data_b = {}
        # data_b['obstacles_vertices'] = torch.tensor(data['obstacles_vertices'], dtype=torch.float64).unsqueeze(0)
        # data_b['target'] = torch.tensor(data['target'], dtype=torch.float64).unsqueeze(0)
        smoothness, curvature_score = path_smoothness(path_numpy)
        # path_torch_b = torch.tensor(path_numpy, dtype=torch.float64).unsqueeze(0)
        # visualize_data_batch_paper(data_b, path_torch_b, save_path=f'imgs/InformedRRTstar{index}')
        print('index=', index, 'time=', time_consumption, 'length=', path_length,
              'smoothness=', smoothness, 'curvature=', curvature_score)
        np.save(save_path, path_numpy)
    else:
        print("No valid path found.")


    return None

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

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(SAVE_DIR, exist_ok=True)
    # test_single(28085)
    test_single(74)
    # save_path_data(0, save_dir)
    # # test_single(28085)
    # if not os.path.exists(RESULT_DIR):
    #     os.makedirs(RESULT_DIR)
    #     print(f"Created directory: {RESULT_DIR}")
    # indices = range(0, 10000)
    # with multiprocessing.Pool(processes=30) as pool:
    #     pool.map(save_path_data, indices)
    # print(f"Starting multiprocessing pool with {25} processes...")
    # with multiprocessing.Pool(processes=25) as pool:
    #     results_iterator = pool.imap_unordered(test_single, indices)
        
    #     for _ in tqdm(results_iterator, total=len(indices)):
    #         pass
    
    # print("All tasks completed.")

    # analyze_results()