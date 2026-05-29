import glob
import json
import os
import sys
import time
import math
import numpy as np
from scipy.interpolate import CubicSpline
from mmengine import fileio
import io
from MPC import MPC_Controller
from scipy.interpolate import splprep, splev
# ==============================================================================
# -- 1. 环境配置与库导入 ---------------------------------------------------------
# ==============================================================================

# 自动寻找并添加 carla .egg 文件
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# 添加 agents 模块路径 (为了使用 PID 控制器)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
from scipy.ndimage import uniform_filter1d
import carla
from agents.navigation.controller import VehiclePIDController

RENDER = True
RESULT_DIR = None
PATH_LOG_DIR = None
ENVS_DIR = '/home/qian/dataset_V7/'
MAX_SPEED = 7.2
MIN_SPEED = 3.6
MAX_LAT_ACCEL = 2.0
# ==============================================================================
# -- 2. 核心辅助类：伪造 Waypoint、 评价体系 ---------------------------------------
# ==============================================================================
class MiniWaypoint:
    def __init__(self, location, rotation=carla.Rotation()):
        self.transform = carla.Transform(location, rotation)
        
import matplotlib.pyplot as plt

class TrajectoryEvaluator:
    def __init__(self):
        # 数据容器
        self.history_cte = []          # 横向跟踪误差 (Cross Track Error)
        self.history_heading_err = []  # 航向误差 (Heading Error)
        self.history_steer = []        # 方向盘转角 (Steering Angle)
        
        self.total_steps = 0
        
        # 优化：记录上一次最近点的索引，实现局部搜索，避免 O(N) 全局搜索
        self.last_closest_idx = 0 

    def compute_step_metrics(self, vehicle, dense_path, dense_yaws):
        """
        在每一帧仿真循环中调用此函数。
        
        :param vehicle: CARLA 车辆对象 (Actor)
        :param dense_path: 完整的密集路径点列表 [carla.Location, ...]
        :param dense_yaws: 对应的密集航向列表 [float (degrees), ...]
        """
        # 1. 获取车辆当前状态
        v_trans = vehicle.get_transform()
        v_loc = v_trans.location
        # 注意：CARLA 的 yaw 是 degrees，这里转为 radians 方便计算
        v_yaw_rad = math.radians(v_trans.rotation.yaw)

        # 2. --- 核心修正：寻找“真·最近点” (Nearest Point) ---
        # 搜索范围：从上一次找到的点往后找 100 个点 (假设车是往前开的)
        search_start = self.last_closest_idx
        search_end = min(self.last_closest_idx + 100, len(dense_path))
        
        min_dist = float('inf')
        closest_idx = self.last_closest_idx

        # 局部搜索
        for i in range(search_start, search_end):
            p = dense_path[i]
            # 计算欧氏距离 (忽略 Z 轴差异，只看平面误差)
            d = math.sqrt((v_loc.x - p.x)**2 + (v_loc.y - p.y)**2)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        # 更新索引缓存，下一次从这里开始找
        self.last_closest_idx = closest_idx
        
        # 3. 记录真实的 CTE (最小距离)
        self.history_cte.append(min_dist)

        # 4. 计算航向误差 (Heading Error)
        # 获取最近点的理想航向 (注意 dense_yaws 存的是 degrees)
        path_yaw_rad = math.radians(dense_yaws[closest_idx])
        
        diff_yaw = abs(v_yaw_rad - path_yaw_rad)
        # 归一化到 [0, pi]
        diff_yaw = diff_yaw % (2 * math.pi)
        if diff_yaw > math.pi:
            diff_yaw = (2 * math.pi) - diff_yaw
        # 存入度数
        self.history_heading_err.append(math.degrees(diff_yaw))

        # 5. --- 补回：记录方向盘转角 ---
        # 这对于计算“控制平滑度”至关重要
        control = vehicle.get_control()
        self.history_steer.append(control.steer)

        self.total_steps += 1

    def get_final_scores(self, feasibility_threshold=0.5):
        """
        实验结束后调用，返回最终指标字典。
        
        :param feasibility_threshold: 判定“可行”的误差阈值 (单位: 米)，默认 0.5m
        """
        if self.total_steps == 0:
            print("⚠️ 警告：没有记录到任何数据！")
            return None

        # 转为 numpy 数组方便计算
        cte_arr = np.array(self.history_cte)
        heading_arr = np.array(self.history_heading_err)
        steer_arr = np.array(self.history_steer)

        # 1. RMSE CTE (均方根横向误差) - 衡量整体跟踪精度
        rmse_cte = np.sqrt(np.mean(cte_arr ** 2))
        
        # 2. Max CTE (最大横向误差) - 衡量最坏情况 (通常发生在急弯)
        max_cte = np.max(cte_arr)

        # 3. Avg Heading Error (平均航向误差)
        avg_heading = np.mean(heading_arr)

        # 4. --- 新增：Dynamic Feasibility Ratio (动态可行性比率) ---
        # 计算有多少帧的误差小于阈值 (例如 0.3m)
        feasible_count = np.sum(cte_arr < feasibility_threshold)
        feasibility_ratio = (feasible_count / self.total_steps) * 100.0

        # 5. Steering Smoothness (控制平滑度)
        # 计算方向盘转角的一阶差分 (变动幅度) 的绝对值之和
        if len(steer_arr) > 1:
            steer_diff = np.diff(steer_arr)
            # 平均每一步方向盘变动了多少 (数值越小越丝滑)
            smoothness_score = np.sum(np.abs(steer_diff)) / (self.total_steps - 1)
        else:
            smoothness_score = 0.0

        return {
            "RMSE_CTE (m)": round(rmse_cte, 4),
            "Max_CTE (m)": round(max_cte, 4),
            "Avg_Heading_Err (deg)": round(avg_heading, 4),
            "Control_Smoothness": round(smoothness_score, 5)
        }

    def plot_results(self):
        """
        画图：生成论文可用的分析图
        """
        if self.total_steps == 0:
            return

        plt.figure(figsize=(12, 8))
        
        # 子图 1: 横向跟踪误差
        plt.subplot(3, 1, 1)
        plt.plot(self.history_cte, label='Cross Track Error (CTE)', color='red', linewidth=1.5)
        # 画一条阈值红线，展示 Feasibility
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Feasibility Threshold (0.3m)')
        plt.title('Tracking Accuracy: Cross Track Error')
        plt.ylabel('Error (m)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')

        # 子图 2: 航向误差
        plt.subplot(3, 1, 2)
        plt.plot(self.history_heading_err, label='Heading Error', color='green', linewidth=1.5)
        plt.title('Tracking Stability: Heading Error')
        plt.ylabel('Error (deg)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')

        # 子图 3: 方向盘转角 (展示平滑度)
        plt.subplot(3, 1, 3)
        plt.plot(self.history_steer, label='Steering Input', color='blue', linewidth=1.5)
        plt.title('Control Smoothness: Steering Input')
        plt.ylabel('Steer (-1 to 1)')
        plt.xlabel('Simulation Steps')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

# ==============================================================================
# -- 3. 辅助函数 ---------------------------------------------------------------
# ==============================================================================
def opendata(path):
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff, allow_pickle=True)
    return npz_data

def xy2xy_heading(xy):
    """
    输入: xy: (N, 2) array
    输出: xy_heading: (N, 3) array, 包含 (x, y, heading)
    策略: 
        - 中间点: Central Difference (P_next - P_prev)
        - 起点: Forward Difference (P_1 - P_0)
        - 终点: Backward Difference (P_last - P_last-1)
    """
    xy = np.array(xy)
    N = xy.shape[0]
    
    if N < 2:
        raise ValueError("Points must contain at least 2 points to calculate heading.")

    # 初始化 heading 数组
    headings = np.zeros(N)
    
    # --- 1. 中间点处理 (Vectorized Central Difference) ---
    # 对应 i 从 1 到 N-2
    # dx = x[i+1] - x[i-1]
    # 利用切片: xy[2:, 0] 是 x[i+1], xy[:-2, 0] 是 x[i-1]
    dx_mid = xy[2:, 0] - xy[:-2, 0]
    dy_mid = xy[2:, 1] - xy[:-2, 1]
    
    headings[1:-1] = np.arctan2(dy_mid, dx_mid)
    
    # --- 2. 边界点处理 ---
    # 起点: 指向第二个点
    dx_start = xy[1, 0] - xy[0, 0]
    dy_start = xy[1, 1] - xy[0, 1]
    headings[0] = np.arctan2(dy_start, dx_start)
    
    # 终点: 由倒数第二个点指向它
    dx_end = xy[-1, 0] - xy[-2, 0]
    dy_end = xy[-1, 1] - xy[-2, 1]
    headings[-1] = np.arctan2(dy_end, dx_end)

    # 拼接结果 (N, 3)
    xy_heading = np.hstack((xy, headings[:, np.newaxis]))
    
    return xy_heading

def smooth_path(waypoints, smoothing_factor=0.5):
    """
    waypoints: np.array shape (N, 2) [[x1, y1], [x2, y2], ...]
    """
    # 1. 去重：RRT*有时会在原地产生重复点，导致插值报错
    # 简单的去重逻辑：如果两点距离太近就扔掉
    diff = np.diff(waypoints, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    valid_indices = np.where(dist > 0.01)[0] + 1
    valid_indices = np.insert(valid_indices, 0, 0) # 加上起点
    clean_waypoints = waypoints[valid_indices]

    # 点太少没法做样条插值 (至少需要 k+1 个点, k通常为3)
    if len(clean_waypoints) < 4:
        return clean_waypoints 

    # 2. B-Spline 插值
    # tck 是样条参数, u 是参数化坐标
    # s 是平滑因子，s=0 表示强制过所有点（可能会抖），s越大越平滑但会偏离原路径
    try:
        tck, u = splprep(clean_waypoints.T, s=smoothing_factor) 
        
        # 3. 生成更密集的点用于跟踪 (比如生成原本点数 5 倍的点)
        u_new = np.linspace(u.min(), u.max(), len(clean_waypoints) * 5)
        x_new, y_new = splev(u_new, tck)
        
        return np.vstack((x_new, y_new)).T
    except Exception as e:
        # 如果插值失败（极其罕见），就返回原路径
        return clean_waypoints

def get_curvature(p1, p2, p3):
    """
    计算三个点(x,y)构成的圆的曲率
    p1, p2, p3: [x, y] format
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y

    # 计算三角形面积 (Shoelace formula)
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    # 计算三边长度
    len_a = np.hypot(x1 - x2, y1 - y2)
    len_b = np.hypot(x2 - x3, y2 - y3)
    len_c = np.hypot(x3 - x1, y3 - y1)
    
    # 防止除以零（直线情况）
    if area < 1e-4:
        return 0.0
        
    # Menger curvature k = 4 * Area / (|a|*|b|*|c|)
    curvature = 4 * area / (len_a * len_b * len_c)
    return curvature

def get_speed(vehicle):
    """
    计算车辆当前的标量速度 (km/h)
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def generate_path(index, method='DiffSlack'):
    path = []
    if method == 'DiffSlack':
        path.append(carla.Location(x=float(0), y=float(0), z=0))
        path_data_file = f'./path/DiffSlack/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
    elif method == 'DC3':
        path.append(carla.Location(x=float(0), y=float(0), z=0))
        path_data_file = f'./path/DC3/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
    elif method == 'IL-Soft':
        path.append(carla.Location(x=float(0), y=float(0), z=0))
        path_data_file = f'./path/IL-Soft/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
    elif method == 'IL_pure':
        path.append(carla.Location(x=float(0), y=float(0), z=0))
        path_data_file = f'./path/IL_pure/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
    elif method == 'DC3-50':
        path.append(carla.Location(x=float(0), y=float(0), z=0))
        path_data_file = f'./path/DC3-50/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
    elif method == 'ENFORCE':
        path.append(carla.Location(x=float(0), y=float(0), z=0))
        path_data_file = f'./path/ENFORCE/batch_{index}.npy'
        path_numpy = np.load(path_data_file, allow_pickle=True)
    return path_numpy

def customize_physics(vehicle):
    """
    修改车辆动力学参数（可选）
    """
    physics_control = vehicle.get_physics_control()
    
    # 修改前轮最大转角为 40 度
    new_wheels = []
    for i, wheel in enumerate(physics_control.wheels):
        if i == 0 or i == 1: # 前轮
            wheel.max_steer_angle = 50.0
        new_wheels.append(wheel)
    
    physics_control.wheels = new_wheels
    physics_control.mass = 1500 # 也可以改质量
    vehicle.apply_physics_control(physics_control)

def get_dynamic_look_ahead(speed_kmh, current_idx, dense_path, dense_yaws):
    # 1. 基础预瞄距离
    base_dist = np.clip(speed_kmh * 0.1, 0.5, 1.0)
    
    # 2. 检测前方多个窗口的航向变化，捕捉连续转弯
    windows = [5, 10, 15]  # 检测前5、10、15个点
    max_diff = 0.0
    
    for window in windows:
        look_ahead_idx = min(current_idx + window, len(dense_path) - 1)
        current_yaw = dense_yaws[current_idx]
        future_yaw = dense_yaws[look_ahead_idx]
        diff = abs(current_yaw - future_yaw)
        if diff > 180:
            diff = 360 - diff
        max_diff = max(max_diff, diff)
    
    # 3. 根据最大航向变化动态缩减预瞄距离
    if max_diff > 30.0:
        # 急弯，预瞄距离缩到最小
        return 0.3
    elif max_diff > 20.0:
        # 中等弯道
        return max(0.3, base_dist * 0.4)
    elif max_diff > 10.0:
        # 轻微弯道
        return max(0.36, base_dist * 0.6)
    
    return base_dist

def smooth_path_linear_interp(sparse_points, resolution=0.1):
    """
    输入稀疏点，输出：
    1. 密集点坐标 (dense_path)
    2. 密集点航向 (dense_yaws)
    3. 密集点曲率 (dense_curvatures)
    """
    x, y = [], []
    for p in sparse_points:
        if isinstance(p, carla.Location):
            x.append(p.x); y.append(p.y)
        else:
            x.append(p[0]); y.append(p[1])

    if len(x) < 2:
        return sparse_points, [], []

    # 去除重复点
    x_clean, y_clean = [x[0]], [y[0]]
    for i in range(1, len(x)):
        dist = math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        if dist > 0.001:
            x_clean.append(x[i])
            y_clean.append(y[i])

    if len(x_clean) < 2:
        return sparse_points, [], []

    # 计算累计弧长
    s = [0]
    for i in range(1, len(x_clean)):
        dist = math.sqrt((x_clean[i] - x_clean[i-1])**2 + (y_clean[i] - y_clean[i-1])**2)
        s.append(s[-1] + dist)
    total_length = s[-1]

    # 生成等间距密集参数
    s_dense = np.arange(0, total_length + resolution, resolution)
    s_dense = s_dense[s_dense <= total_length]

    # 线性插值坐标
    x_dense = np.interp(s_dense, s, x_clean)
    y_dense = np.interp(s_dense, s, y_clean)

    # 计算航向：用相邻点的方向
    dx = np.diff(x_dense, append=x_dense[-1] - x_dense[-2] + x_dense[-1])
    dy = np.diff(y_dense, append=y_dense[-1] - y_dense[-2] + y_dense[-1])
    yaw_dense = np.arctan2(dy, dx)

    # 计算曲率：用三点法
    k_dense = np.zeros(len(x_dense))
    for i in range(1, len(x_dense) - 1):
        x1, y1 = x_dense[i-1], y_dense[i-1]
        x2, y2 = x_dense[i], y_dense[i]
        x3, y3 = x_dense[i+1], y_dense[i+1]
        a = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        b = math.sqrt((x3-x2)**2 + (y3-y2)**2)
        c = math.sqrt((x3-x1)**2 + (y3-y1)**2)
        area = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2.0
        denom = max(a * b * c, 1e-6)
        k_dense[i] = 2.0 * area / denom
        # 加符号
        cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        if cross < 0:
            k_dense[i] = -k_dense[i]
    k_dense[0] = k_dense[1]
    k_dense[-1] = k_dense[-2]

    # 对曲率做滑动平均平滑
    from scipy.ndimage import uniform_filter1d
    k_dense = uniform_filter1d(k_dense, size=10)

    # 封装返回
    dense_path_objects = []
    dense_yaw_degrees = []
    z_val = sparse_points[0].z if isinstance(sparse_points[0], carla.Location) else 0
    for i in range(len(x_dense)):
        loc = carla.Location(x=float(x_dense[i]), y=float(y_dense[i]), z=float(z_val))
        dense_path_objects.append(loc)
        dense_yaw_degrees.append(math.degrees(yaw_dense[i]))

    return dense_path_objects, dense_yaw_degrees, k_dense

def calculate_feedforward(curvature, wheelbase=2.87, max_steer_deg=40.0):
    """
    根据曲率计算前馈转向指令
    :param curvature: 当前路径点的曲率
    :param wheelbase: 轴距 (米)
    :param max_steer_deg: 车辆物理最大转角 (度)
    :return: 归一化的转向指令 [-1, 1]
    """
    # 1. 运动学自行车模型公式: delta = arctan(L * k)
    # 结果是弧度
    steer_rad = math.atan(wheelbase * curvature)
    
    # 2. 将弧度转换为 CARLA 的控制比例 [-1, 1]
    max_steer_rad = math.radians(max_steer_deg)
    steer_norm = steer_rad / max_steer_rad
    
    # 3. 限制范围 (防止计算出的理论值超过机械极限)
    return np.clip(steer_norm, -1.0, 1.0)

# ==============================================================================
# -- 4. 碰撞检测 ----------------------------------------------------------------
# ==============================================================================
def get_rect_points(xy_heading, width=1.9, length=4.7):
    '''
    NumPy 向量化版本
    输入: xy_heading: (N, 3) array, 包含 (x, y, heading)
    输出: rectangles: (N, 4, 2) array, 包含每个矩形的4个顶点坐标
    '''
    
    # 确保是 numpy array
    if not isinstance(xy_heading, np.ndarray):
        xy_heading = np.array(xy_heading)
        
    N = xy_heading.shape[0]
    heading = xy_heading[:, 2]
    
    # 1. 计算三角函数 (N, 1)
    # 使用 keepdims=True 或 np.newaxis 为了后续广播
    cos_h = np.cos(heading)[:, np.newaxis] 
    sin_h = np.sin(heading)[:, np.newaxis] 

    # 2. 定义局部坐标系下的四个顶点 (4, 2)
    # 顺序: 右前, 右后, 左后, 左前 (与你的 PyTorch 版本一致)
    half_l = length / 2.0
    half_w = width / 2.0
    
    # 这里我们将 x 和 y 的偏移量分开定义，方便向量化计算
    # dx: (1, 4), dy: (1, 4)
    dx = np.array([half_l, half_l, -half_l, -half_l])
    dy = np.array([half_w, -half_w, -half_w, half_w])

    # 3. 向量化旋转计算
    # 公式: 
    # x' = x * cos - y * sin
    # y' = x * sin + y * cos
    # 利用广播: (N, 1) * (1, 4) -> (N, 4)
    
    rot_x = dx * cos_h - dy * sin_h  # shape: (N, 4)
    rot_y = dx * sin_h + dy * cos_h  # shape: (N, 4)

    # 4. 组合旋转后的偏移量 (N, 4, 2)
    rotated_offsets = np.stack((rot_x, rot_y), axis=-1)

    # 5. 加上中心点坐标
    # centers shape: (N, 2) -> (N, 1, 2) 用于广播
    centers = xy_heading[:, :2][:, np.newaxis, :]
    
    rectangles = centers + rotated_offsets
    
    return rectangles

def check_containment(poly1, poly2):
    """检查一个四边形是否完全包含在另一个中"""
    # 检查poly1的所有顶点是否都在poly2内
    if all(point_in_polygon(poly1[:,i], poly2) for i in range(4)):
        return True
    
    # 检查poly2的所有顶点是否都在poly1内
    if all(point_in_polygon(poly2[:,i], poly1) for i in range(4)):
        return True
    
    return False

def point_in_polygon(point, polygon):
    """使用射线法判断点是否在四边形内"""
    x, y = point
    n = 4  # 四边形有4个顶点
    inside = False
    
    px, py = polygon[0, 0], polygon[1, 0]
    for i in range(n + 1):
        qx, qy = polygon[0, i % n], polygon[1, i % n]
        if y > min(py, qy):
            if y <= max(py, qy):
                if x <= max(px, qx):
                    if py != qy:
                        xinters = (y - py) * (qx - px) / (qy - py) + px
                    if px == qx or x <= xinters:
                        inside = not inside
        px, py = qx, qy
    
    return inside

def check_edges_intersection(poly1, poly2):
    """检查两个四边形的边是否有相交"""
    for i in range(4):
        p1 = poly1[:, i]
        p2 = poly1[:, (i+1)%4]
        
        for j in range(4):
            p3 = poly2[:, j]
            p4 = poly2[:, (j+1)%4]
            
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def segments_intersect(a1, a2, b1, b2):
    """检查两条线段是否相交"""
    # 确保所有点都是二维坐标
    a1 = np.asarray(a1).flatten()[:2]
    a2 = np.asarray(a2).flatten()[:2]
    b1 = np.asarray(b1).flatten()[:2]
    b2 = np.asarray(b2).flatten()[:2]
    
    # 使用叉积方法判断线段相交
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    # 检查一般情况下的相交
    case1 = ccw(a1, b1, b2) != ccw(a2, b1, b2)
    case2 = ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    if case1 and case2:
        return True
    
    # 检查端点重合情况
    if (np.array_equal(a1, b1) or np.array_equal(a1, b2) or 
        np.array_equal(a2, b1) or np.array_equal(a2, b2)):
        return True
    
    # 检查共线重叠
    if is_point_on_segment(a1, b1, b2) or is_point_on_segment(a2, b1, b2):
        return True
    if is_point_on_segment(b1, a1, a2) or is_point_on_segment(b2, a1, a2):
        return True
    
    return False

def is_point_on_segment(p, a, b):
    """检查点p是否在线段ab上"""
    p = np.asarray(p).flatten()[:2]
    a = np.asarray(a).flatten()[:2]
    b = np.asarray(b).flatten()[:2]
    
    cross = (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])
    if not np.isclose(cross, 0, atol=1e-8):
        return False
    
    min_x = min(a[0], b[0])
    max_x = max(a[0], b[0])
    min_y = min(a[1], b[1])
    max_y = max(a[1], b[1])
    
    return (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y)

def check_polygon_intersection(poly1, poly2):
    """
    检查两个四边形是否相交
    参数:
        poly1, poly2: 2x4 numpy数组，表示四边形的顶点坐标
                     格式为 [[x1,x2,x3,x4], [y1,y2,y3,y4]]
    返回:
        bool: 如果相交返回True，否则返回False
    """
    # 验证输入形状
    if poly1.shape == (4,2):
        poly1 = poly1.T
    if poly2.shape == (4,2):
        poly2 = poly2.T
    
    if poly1.shape != (2,4) or poly2.shape != (2,4):
        raise ValueError("输入必须是2x4的numpy数组")
    
    # 检查边相交情况
    if check_edges_intersection(poly1, poly2):
        return True
    
    # 检查包含关系
    if check_containment(poly1, poly2):
        return True
    
    return False

def collision_check(path, obstacles):
    """
    检查路径与障碍物之间的碰撞
    参数:
        path: (N, 3) array, 包含 (x, y, heading)
        obstacles: (M, 4, 2) numpy arrays, 每个数组表示一个障碍物的四个顶点
    返回:
        bool: 如果发生碰撞返回True 否则返回False
    """
    # 获取路径的矩形表示
    path_rects = get_rect_points(path)

    # 检查每个路径矩形是否与任何障碍物相交
    for rect in path_rects:
        for obs in obstacles:
            if check_polygon_intersection(rect, obs):
                return True
    return False

# ==============================================================================
# -- 5. 主函数 -----------------------------------------------------------------
# ==============================================================================
def main(index, methods='hard'):
    actor_list = []
    evaluator = TrajectoryEvaluator()
    result_save_file = f'{RESULT_DIR}/{index}.json'
    if os.path.exists(result_save_file):
        print(f"⚠️ 结果文件 {result_save_file} 已存在，跳过此轮测试")
        return
    actual_path = []
    try:
        # 生成路径
        path_interval = 0.2
        path_points_numpy = generate_path(index, methods)
        path_points = []
        for point in path_points_numpy:
            path_points.append(carla.Location(x=float(point[0]), y=float(point[1]), z=0))
        path_points, dense_yaws, dense_curvatures = smooth_path_linear_interp(path_points, resolution=path_interval)
        begin_point = path_points[5]
        begin_yaw = math.degrees(math.atan2(begin_point.y, begin_point.x))
        # --- A. 连接服务器 ---
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        
        # 推荐：设置同步模式 (Synchronous Mode)，这样 PID 控制最稳定
        # 如果你不想用同步模式，可以把这块注释掉，但在低帧率下 PID 可能会抖动
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = not RENDER
        settings.fixed_delta_seconds = 0.02 # 20 FPS 的物理计算频率
        world.apply_settings(settings)
        # --- B. 生成车辆 ---
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # 强制在 (0, 0, 2) 生成
        spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2.0), carla.Rotation(yaw=begin_yaw))
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        
        if not vehicle:
            print("❌ 生成失败，位置可能被占用")
            return
        
        actor_list.append(vehicle)
        # print("✅ 车辆已生成，准备初始化...")

        # --- C. 刹车等待 (防止落地弹跳导致位移) ---
        # 先拉手刹
        vehicle.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        mpc = MPC_Controller(vehicle)
        # 让物理引擎空转 20 帧 (约1秒)，等车稳住
        for _ in range(20):
            world.tick()

        # 修改物理参数
        customize_physics(vehicle)

        # --- D. 初始化 PID 控制器 ---
        # 这里的参数 K_P, K_D, K_I 是可以调的。
        # 横向控制 (方向盘): P=1.0 比较生硬，P=0.5 比较软。
        pid_controller = VehiclePIDController(vehicle,
                                        args_lateral={'K_P': 3.0, 'K_D': 1.0, 'K_I': 0.0},
                                        args_longitudinal={'K_P': 0.8, 'K_D': 0.0, 'K_I': 0.0})

        
        # print(f"🏁 开始跟踪路径，总点数: {len(path_points)}")
        FF_GAIN = 1.0  # 前馈增益，原来是1.2*1.2
        last_closest_idx = 0

        while True:
            world.tick()
            
            current_loc = vehicle.get_location()
            v_trans = vehicle.get_transform()
            v_yaw = math.radians(v_trans.rotation.yaw)
            actual_path.append([current_loc.x, current_loc.y, v_yaw])

            # 动态搜索范围
            speed = get_speed(vehicle)
            search_range = max(50, int(speed * 2))
            start_idx = last_closest_idx
            end_idx = min(last_closest_idx + search_range, len(path_points))

            min_dist = float('inf')
            current_closest_idx = last_closest_idx

            for i in range(start_idx, end_idx):
                dist = vehicle.get_location().distance(path_points[i])
                if dist < min_dist:
                    min_dist = dist
                    current_closest_idx = i

            # 防止回退
            last_closest_idx = max(last_closest_idx, current_closest_idx)

            # 预瞄距离
            look_ahead_dist = get_dynamic_look_ahead(speed, last_closest_idx, path_points, dense_yaws)
            look_ahead_step = int(look_ahead_dist / path_interval)

            target_idx = last_closest_idx + look_ahead_step
            if target_idx >= len(path_points):
                target_idx = len(path_points) - 1

            target_loc = path_points[target_idx]
            target_yaw = dense_yaws[target_idx]

            if current_closest_idx >= len(path_points) - 10:
                break

            target_loc.z = current_loc.z

            # 统一使用预计算曲率
            curvature_idx = min(target_idx, len(dense_curvatures) - 1)
            k_current = dense_curvatures[curvature_idx]

            # 限速计算
            v_limit_ms = np.sqrt(MAX_LAT_ACCEL / (abs(k_current) + 1e-6))
            v_limit_kmh = v_limit_ms * 3.6
            target_speed = np.clip(v_limit_kmh, MIN_SPEED, MAX_SPEED)

            fake_waypoint = MiniWaypoint(target_loc, carla.Rotation(yaw=target_yaw))
            control = pid_controller.run_step(target_speed, fake_waypoint)
            pid_steer = control.steer
            
            k_current = np.clip(dense_curvatures[curvature_idx], -0.25, 0.25)
            # 前馈计算，统一增益
            ff_steer = calculate_feedforward(k_current, wheelbase=2.87, max_steer_deg=45.0)
            direction_fix = 1.0
            final_steer = pid_steer + ff_steer * direction_fix * FF_GAIN

            control.steer = np.clip(final_steer, -1.0, 1.0)
            vehicle.apply_control(control)
            
            # 3. 视觉调试：画出目标点和路径
            if RENDER:
                # 画个红色的 X 在目标点
                world.debug.draw_point(target_loc, size=0.1, color=carla.Color(255, 0, 0), life_time=0.1)
                for point in path_points:
                    world.debug.draw_point(point, size=0.05, color=carla.Color(0, 0, 255), life_time=0.1)
                # 画出车辆位置
                world.debug.draw_point(current_loc, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)

                # 4. 更新相机位置 (上帝视角跟车)
                spectator = world.get_spectator()
                transform = vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30),carla.Rotation(pitch=-90)))
            if last_closest_idx > 10: # 避免一开始就评估，等车稳定跟踪后再评估
                evaluator.compute_step_metrics(vehicle, path_points, dense_yaws)
                if evaluator.history_cte[-1] > 1.5:
                    print(f"⚠️ 车辆偏离过大，当前CTE: {evaluator.history_cte[-1]:.2f}m，提前结束评估")
                    break

        scores = evaluator.get_final_scores()
        print(f"------ 动力学可行性评估报告_{index} ------")
        print(scores)
        world.apply_settings(original_settings)
        # if RENDER:
        #     evaluator.plot_results()

    finally:
        # print("🧹 正在清理...")
        # 清除绘制的障碍物
        for actor in actor_list:
            actor.destroy()
        print("👋 完成。")
    
    with open(result_save_file, 'w') as f:
        json.dump(scores, f)
    return

def analyze_results():
    json_files = glob.glob(os.path.join(RESULT_DIR, '*.json'))
    save_file = os.path.join(RESULT_DIR, 'summary.json')
    total_cases = len(json_files)
    if total_cases == 0:
        print("No result files found!")
        return
    results = {
        "RMSE_CTE (m)": [],
        "Max_CTE (m)": [],
        "Avg_Heading_Err (deg)": [],
        "Feasibility_Ratio (%)": [],
        "Control_Smoothness": []
    }
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                for key in results.keys():  # ← 移到 try 里面
                    if key in data:
                        results[key].append(data[key])
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {json_file}")
    # 计算每个指标的平均值
    for key in results.keys():
        if results[key]:
            results[key] = np.mean(results[key])
    print(f"------ 动力学可行性评估报告 ------")
    print(results)
    with open(save_file, 'w') as f:
        json.dump(results, f)
    
if __name__ == '__main__':
    RENDER = False
    alg = 'DiffSlack' # 'DiffSlack', 'DC3', 'IL-Soft' 'IL_pure' 'DC3-50' 'ENFORCE'
    RESULT_DIR = f'res/{alg}'
    os.makedirs(RESULT_DIR, exist_ok=True)
    indexs = range(0, 100)
    for i in indexs:
        main(i, alg)
    analyze_results()
