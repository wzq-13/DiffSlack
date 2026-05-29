import numpy as np
from typing import Callable, Dict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import globalvar
from utils.prob import xy2xy_heading
import os
from others.PlanHybridAStarPath import CreateVehiclePolygon
from shapely.ops import unary_union
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, CAP_STYLE, JOIN_STYLE
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

planning_scale_ = globalvar.planning_scale_
hybrid_astar_ = globalvar.hybrid_astar_
Nobs = globalvar.Nobs
vehicle_TPBV_ = globalvar.vehicle_TPBV_
vehicle_geometrics_ = globalvar.vehicle_geometrics_
vehicle_kinematics_ = globalvar.vehicle_kinematics_
margin_obs_ = globalvar.margin_obs_ 

# --- 全局绘图风格设置 (建议放在代码最开头) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # IEEE 偏好字体
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['mathtext.fontset'] = 'stix' # 公式字体风格


def path_clean(path, target):
    """
    清理路径，去除重复点和距离过近的点
    """
    # # 找到离终点最近的点
    dists_to_target = np.linalg.norm(path[:, :2] - target, axis=1)
    min_index = np.argmin(dists_to_target)
    path = path[:min_index]
    # # print(path)

    i = 1
    while i < len(path):
        min_index = i-1
        min_dist = np.linalg.norm(path[i, :2] - path[min_index, :2])
        for j in range(0, i-1):
            dist = np.linalg.norm(path[j, :2] - path[i, :2])
            if dist < min_dist:
                min_dist = dist
                min_index = j
        if min_index != i-1:
            # 删掉从 min_index+1 到 i-1 的点
            path = np.delete(path, np.s_[min_index+1:i], axis=0)
            i = min_index + 1
        else:
            i += 1
    # return path
    cleaned_path = [path[0]]
    end_point = path[-1]
    i=1
    while i < len(path)-1:
        next_p = None
        next_index = i
        for j in range(i, len(path)-1):
            dist = np.linalg.norm(path[j, :2] - cleaned_path[-1][:2])
            if dist >= 0.5:  # 保留距离大于阈值的点
                next_p = path[j]
                next_index = j
                break
        if next_p is not None:
            if np.linalg.norm(end_point[:2] - next_p[:2]) < 0.5:
                break
            cleaned_path.append(next_p)
        i = next_index if next_index > i else i + 1  # 防止死循环
    cleaned_path.append(path[-1])  # 确保终点被添加
    return np.array(cleaned_path)

def xy2xy_heading_numpy(xy):
    """
    将 (x,y) 转换为 (x,y,heading)
    - 首点：前向差分
    - 中间点：中心差分（前后点）
    - 末点：后向差分
    xy: shape (N, 2)
    返回: shape (N, 3)
    """
    headings = np.zeros(len(xy))

    # 首点：前向差分
    dx = xy[1, 0] - xy[0, 0]
    dy = xy[1, 1] - xy[0, 1]
    headings[0] = np.arctan2(dy, dx)

    # 中间点：中心差分
    dx = xy[2:, 0] - xy[:-2, 0]
    dy = xy[2:, 1] - xy[:-2, 1]
    headings[1:-1] = np.arctan2(dy, dx)

    # 末点：后向差分
    dx = xy[-1, 0] - xy[-2, 0]
    dy = xy[-1, 1] - xy[-2, 1]
    headings[-1] = np.arctan2(dy, dx)

    return np.hstack([xy, headings.reshape(-1, 1)])

from shapely.geometry import Polygon
from shapely.ops import unary_union

def get_swept_path_as_polygon(traj_full, width, length, step=2):
    """
    使用几何并集生成完美的连续扫过区域
    :param traj_full: 轨迹点 [N, 3] (x, y, theta)
    :param width: 车宽
    :param length: 车长
    :param step: 采样步长（为了性能，没必要每个点都算，每隔几个点算一个即可）
    :return: shapely.geometry.Polygon 或 MultiPolygon 对象
    """
    if len(traj_full) < 2:
        return None

    polys = []
    # 半长和半宽
    hl = length / 2.0
    hw = width / 2.0
    
    # 稍微给一点点 buffer 避免浮点数计算时的微小缝隙，虽然 Union 通常能处理
    # 但在极度紧密的点集中，微小的膨胀有助于融合
    
    # 降采样遍历轨迹点
    for i in range(0, len(traj_full), step):
        x, y, theta = traj_full[i]
        c, s = np.cos(theta), np.sin(theta)
        
        # 计算四个角点 (顺时针: FL, FR, RR, RL)
        # 注意：这里根据你的坐标系调整，假设 x向前, y向左
        corners = np.array([
            [x + hl*c - hw*s, y + hl*s + hw*c], # FL
            [x + hl*c + hw*s, y + hl*s - hw*c], # FR
            [x - hl*c + hw*s, y - hl*s - hw*c], # RR
            [x - hl*c - hw*s, y - hl*s + hw*c]  # RL
        ])
        polys.append(Polygon(corners))

    # 核心魔法：将所有矩形融合成一个多边形
    swept_shape = unary_union(polys)
    
    return swept_shape

def generate_swept_contour_precise(traj_full, width, length):
    """
    精确生成车辆扫掠体轮廓
    原理：连接相邻车辆矩形的对应角点，形成连续边界
    """
    if len(traj_full) < 2:
        return None
    
    # 1. 生成每个点的矩形角点
    rectangles = []
    for i in range(len(traj_full)):
        x, y, theta = traj_full[i]
        half_l = length / 2
        half_w = width / 2
        
        # 车辆四个角点（顺时针）
        corners = np.array([
            [x + half_l * np.cos(theta) - half_w * np.sin(theta),  # 右前
             y + half_l * np.sin(theta) + half_w * np.cos(theta)],
            [x + half_l * np.cos(theta) + half_w * np.sin(theta),  # 左前
             y + half_l * np.sin(theta) - half_w * np.cos(theta)],
            [x - half_l * np.cos(theta) + half_w * np.sin(theta),  # 左后
             y - half_l * np.sin(theta) - half_w * np.cos(theta)],
            [x - half_l * np.cos(theta) - half_w * np.sin(theta),  # 右后
             y - half_l * np.sin(theta) + half_w * np.cos(theta)]
        ])
        rectangles.append(corners)
    
    rectangles = np.array(rectangles)  # [n, 4, 2]
    
    # 2. 构建扫掠体外边界
    n = len(rectangles)
    
    # 左边界：连接所有矩形的左前角到左后角
    left_boundary = []
    for i in range(n):
        left_boundary.append(rectangles[i, 1])  # 左前角
    for i in range(n-1, -1, -1):
        left_boundary.append(rectangles[i, 2])  # 左后角（反向）
    
    # 右边界：连接所有矩形的右后角到右前角
    right_boundary = []
    for i in range(n):
        right_boundary.append(rectangles[i, 3])  # 右后角
    for i in range(n-1, -1, -1):
        right_boundary.append(rectangles[i, 0])  # 右前角（反向）
    
    # 合并边界形成多边形
    swept_contour = np.vstack([left_boundary, right_boundary, left_boundary[0:1]])
    
    return swept_contour

def densify_trajectory_smooth(traj, max_step=0.05, method='cubic'):
    """
    使用高阶样条插值加密轨迹，使其更加平滑自然
    :param traj: [N, 3] numpy array (x, y, theta)
    :param max_step: 插值最大步长
    :param method: 'cubic' (最平滑) 或 'akima' (防过冲，更安全)
    :return: 加密后的轨迹 [M, 3]
    """
    if len(traj) < 3:
        # 点太少无法进行样条插值，回退到线性
        from scipy.interpolate import interp1d
        # ... (可以使用你原来的逻辑作为 fallback) ...
        return traj 

    x = traj[:, 0]
    y = traj[:, 1]
    yaw = traj[:, 2]

    # 1. 计算路径累积距离 (Arc Length)
    # 这是插值的自变量 x
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dist[-1]

    # 2. 生成新的采样点距离
    num_points = int(total_dist / max_step) + 1
    new_dists = np.linspace(0, total_dist, num_points)
    
    # 3. 选择插值器
    # CubicSpline: 最平滑，曲率连续，但在急转弯处可能会“甩尾” (Overshoot)
    # Akima1DInterpolator: 平滑度稍差，但更稳定，不会产生剧烈的震荡
    if method == 'cubic':
        Interpolator = CubicSpline
    else:
        Interpolator = Akima1DInterpolator

    # 4. 插值 X 和 Y
    # bc_type='natural' 表示两端曲率为0 (自然样条)，通常效果最好
    f_x = Interpolator(cum_dist, x)
    f_y = Interpolator(cum_dist, y)
    new_x = f_x(new_dists)
    new_y = f_y(new_dists)

    # 5. 插值 Yaw (关键处理)
    # 必须先解缠绕 (Unwrap)，否则在 -pi/pi 处会插值出错误的数据
    yaw_unwrapped = np.unwrap(yaw)
    f_yaw = Interpolator(cum_dist, yaw_unwrapped)
    new_yaw = f_yaw(new_dists)
    
    # 可选：重新归一化到 -pi ~ pi
    # new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

    return np.column_stack((new_x, new_y, new_yaw))

def get_smooth_swept_path(traj_full, width, length, max_step=0.1, smooth_radius=0.2):
    """
    生成平滑的扫过区域
    :param max_step: 插值密度，建议 0.05 - 0.1
    :param smooth_radius: 平滑倒角半径，值越大边缘越圆润，建议 0.1 - 0.3
    """
    # 1. 第一步：加密轨迹
    # dense_traj = densify_trajectory_smooth(traj_full, max_step=max_step)
    dense_traj = traj_full
    # 2. 第二步：生成高密度矩形
    polys = []
    hl = length / 2.0
    hw = width / 2.0
    
    # 向量化计算以提高性能
    x = dense_traj[:, 0]
    y = dense_traj[:, 1]
    theta = dense_traj[:, 2]
    c = np.cos(theta)
    s = np.sin(theta)
    
    # 预计算四个角的偏移量
    # FL, FR, RR, RL
    dx = np.array([hl, hl, -hl, -hl])
    dy = np.array([-hw, hw, hw, -hw]) # 注意你的左右定义，这里假设 standard
    
    # 循环生成 Polygon (Shapely 创建对象这一步很难向量化)
    # 这里的性能瓶颈在于 Polygon 对象的创建
    for i in range(len(dense_traj)):
        # 旋转矩阵手动展开
        # global_x = x + local_x * c - local_y * s
        # global_y = y + local_x * s + local_y * c
        corners_x = x[i] + dx * c[i] - dy * s[i]
        corners_y = y[i] + dx * s[i] + dy * c[i]
        
        polys.append(Polygon(np.column_stack((corners_x, corners_y))))

    # 3. 第三步：合并
    raw_union = unary_union(polys)
    
    # 4. 第四步：形态学平滑 (关键!)
    # buffer(r) -> 膨胀并圆角化
    # buffer(-r) -> 腐蚀回原大小
    # join_style=1 (ROUND) 是平滑的关键
    smoothed_shape = raw_union.buffer(smooth_radius, join_style=JOIN_STYLE.round) \
                              .buffer(-smooth_radius, join_style=JOIN_STYLE.round)
    
    return smoothed_shape

def animate_trajectory(traj_full, obstacles_vertices, target_pos, save_path="trajectory_animation.gif"):
    """
    制作车辆行驶动画，包含动态更新的扫掠区域
    :param traj_full: 完整轨迹点 [N, 3]
    :param obstacles_vertices: 障碍物顶点数组
    :param target_pos: 目标点坐标 [x, y]
    :param save_path: 保存路径 (.gif 或 .mp4)
    """
    
    # 1. 创建画布
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 颜色定义 (保持和你之前的风格一致)
    COLOR_SWEPT = '#1f77b4'
    COLOR_TRAJ = '#FF4500'
    COLOR_OBS = '#2F4F4F'        # 障碍物 (深石板灰)
    COLOR_OBS_FILL = "#BEBEBE"
    
    # 2. 绘制静态背景 (障碍物、起点、终点)
    # 障碍物
    if obstacles_vertices is not None:
        polygons = obstacles_vertices.reshape(-1, 4, 2)
        for poly in polygons:
            ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                    linewidth=1.2, hatch='', alpha=1.0, zorder=1)
            
    # 终点
    ax.scatter(target_pos[0], target_pos[1], color='#DC143C', s=200, marker='*', 
               edgecolors='black', zorder=5, label='Target')
    # 起点
    ax.scatter(traj_full[0, 0], traj_full[0, 1], color='#32CD32', s=120, marker='o', 
               edgecolors='black', zorder=5, label='Start')

    # 设置坐标轴 (根据轨迹范围动态调整)
    margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
    margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
    ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
    ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
    ax.set_aspect('equal')
    ax.axis('off') # 关闭坐标轴刻度，看起来更像演示视频

    # 3. 初始化动态元素 (Artists)
    # 扫掠区域 (初始为空)
    swept_patch = None 
    
    # 轨迹线
    traj_line, = ax.plot([], [], color=COLOR_TRAJ, linewidth=2.5, zorder=4)
    
    # 当前车辆 (初始位置)
    current_car_patch = MplPolygon(np.zeros((4, 2)), closed=True, 
                                   fc='none', ec='black', lw=1.5, zorder=6)
    ax.add_patch(current_car_patch)

    # 4. 动画更新函数
    # 提示：为了加快渲染速度，可以跳帧，例如 frames=range(0, len, 2)
    def update(frame_idx):
        nonlocal swept_patch
        
        # 获取当前进度的数据
        current_traj = traj_full[:frame_idx+1]
        current_pose = traj_full[frame_idx]
        
        # --- A. 更新轨迹线 ---
        traj_line.set_data(current_traj[:, 0], current_traj[:, 1])
        
        # --- B. 更新当前车辆姿态 ---
        # 计算当前这一帧的车辆矩形
        rects = get_rect_points_vectorized(
            np.array([current_pose]), 
            width=globalvar.vehicle_geometrics_.vehicle_width, 
            length=globalvar.vehicle_geometrics_.vehicle_length
        )
        current_car_patch.set_xy(rects[0])
        
        # --- C. 更新扫掠区域 (最耗时的部分) ---
        # 如果你觉得动画生成太慢，可以将 max_step 调大一点，或者只在 frame_idx % 5 == 0 时更新区域
        if frame_idx > 1:
            # 清除上一帧的区域
            if swept_patch:
                swept_patch.remove()
                
            # 计算新的累积区域
            # 注意：这里我们传入 current_traj，即只计算走过的路
            poly = get_smooth_swept_path(
                current_traj, 
                width=globalvar.vehicle_geometrics_.vehicle_width, 
                length=globalvar.vehicle_geometrics_.vehicle_length,
                max_step=0.05,    # 动态演示时精度可以稍微低一点以提高速度
                smooth_radius=0.15
            )
            
            # 将 Shapely 多边形转换为 Matplotlib Patch
            if poly and not poly.is_empty:
                if poly.geom_type == 'Polygon':
                    xs, ys = poly.exterior.xy
                    swept_patch = MplPolygon(np.column_stack((xs, ys)), 
                                             fc=COLOR_SWEPT, ec='none', alpha=0.3, zorder=2)
                    ax.add_patch(swept_patch)
                elif poly.geom_type == 'MultiPolygon':
                    # 处理多边形情况略复杂，简单起见我们只取最大的或者合并绘制
                    # 动画中一般轨迹是连续的，很少出现 MultiPolygon
                    pass 

        return traj_line, current_car_patch, swept_patch

    # 5. 生成动画
    # frames: 帧数。如果点太多，建议切片: range(0, len(traj_full), 2)
    print("开始生成动画...")
    ani = animation.FuncAnimation(fig, update, frames=range(0, len(traj_full), 1), 
                                  interval=1, blit=False, repeat=False)
    
    # 6. 保存
    if save_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=10)
        ani.save(save_path, writer=writer)
    elif save_path.endswith('.mp4'):
        # 需要安装 ffmpeg
        writer = animation.FFMpegWriter(fps=20, extra_args=['-vcodec', 'libx264'])
        ani.save(save_path, writer=writer)
        
    print(f"动画已保存至: {save_path}")
    plt.close()

def visualize_data_batch_paper(datas, trajectorys, save_path=None):
    """
    TMECH 投稿用：连续扫掠区域+中心线，高学术可视化质量
    """
    os.makedirs(save_path, exist_ok=True)
    USE_ZOOMED_INSET = False  # 是否使用放大镜效果展示细节
    # --- 高级配色方案 (优化对比度) ---
    COLOR_SWEPT = "#187dc5"      # 轨迹扫掠区域 (主色，调整透明度)
    COLOR_TRAJ = 'blue'       # 轨迹中心线
    COLOR_OBS = "#1E2C2C"        # 障碍物 (深石板灰)
    COLOR_GOAL = '#DC143C'       # 终点 (深红)
    COLOR_START = '#32CD32'      # 起点 (亮绿)
    COLOR_OBS_FILL = "#6E6C6C"   # 障碍物填充 (浅灰)
    
    idxxx = 84
    # for i in range(1):
    for i in range(trajectorys.shape[0]):
        # if i < idxxx:
        #     continue
        data = {key: datas[key][i].cpu().numpy() for key in datas}
        traj_full = trajectorys[i].cpu().detach().numpy()
        # data = datas
        # traj_full = trajectorys
        # traj_full_0 = np.vstack((np.array([[ -1.0, 0.0]]),np.array([[0.0, 0.0]]), traj_full))
        obstacles_vertices = data['obstacles_vertices']
        target = data['target']
        
        # traj_full_clean = path_clean(traj_full, target)
        traj_full_clean = traj_full
        # print('*' * 20)
        # print(traj_full_clean.shape)

        # 前面加上历史点(-1,0) 和起始点(0,0)
        # traj_full = np.vstack((np.array([[ -1.0, 0.0]]),np.array([[0.0, 0.0]]), traj_full))
        # traj_full_clean = np.vstack((np.array([[ -1.0, 0.0]]),np.array([[0.0, 0.0]]), traj_full_clean))
        # 转换为 (x,y,heading)
        traj_full = xy2xy_heading_numpy(traj_full)
        traj_full_clean = xy2xy_heading_numpy(traj_full_clean)
        # traj_full = traj_full[1:]
        
        # animate_trajectory(
        #     traj_full,
        #     obstacles_vertices,
        #     target,
        #     save_path=os.path.join(save_path, f'trajectory_{i}.gif')
        # )
        # continue
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        if USE_ZOOMED_INSET:
            axins = zoomed_inset_axes(ax, zoom=2.0, loc='center', bbox_to_anchor=(0.83, 0.25), bbox_transform=ax.transAxes) 
        
        # 1. 绘制障碍物 (优化视觉层次)
        polygons = obstacles_vertices.reshape(-1, 4, 2)
        for poly in polygons:
            ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                    linewidth=0.5, hatch='', alpha=1.0, zorder=1)
            if USE_ZOOMED_INSET:
                axins.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                           linewidth=0.5, hatch='', alpha=1.0, zorder=1)
        
        # 2. 生成连续扫掠区域 (核心改进)
        # 获取所有路径点的车辆轮廓
        swept_poly = get_smooth_swept_path(
            traj_full_clean, 
            width=1.8, 
            length=globalvar.vehicle_geometrics_.vehicle_length,
            max_step=0.01,    # 足够密，保证False转弯处不出现大断层
            smooth_radius=2.0 # 磨掉锯齿
        )

        if swept_poly is not None:
            # Shapely 可能会返回 Polygon 或 MultiPolygon (如果路径中间断开)
            if swept_poly.geom_type == 'Polygon':
                shapes = [swept_poly]
            elif swept_poly.geom_type == 'MultiPolygon':
                shapes = swept_poly.geoms
            else:
                shapes = []

            for shape in shapes:
                x, y = shape.exterior.xy
                # 技巧：
                # 1. edge color 和 face color 一致，避免内部出现细线
                # 2. alpha 设置低一点，体现“区域”感
                # 3. antialiased=True 开启抗锯齿，边缘更平滑
                ax.fill(x, y, 
                        facecolor=COLOR_SWEPT, 
                        edgecolor=COLOR_SWEPT, # 关键：边缘色同填充色
                        alpha=0.25, 
                        zorder=2,
                        label='Swept Area' if shape == shapes[0] else None) # 只给第一个加标签

                # 进阶美化：如果你想要一个清晰的深色轮廓线，只画最外圈
                ax.plot(x, y, color=COLOR_SWEPT, linewidth=0.8, alpha=0.5, zorder=2)
                
                if USE_ZOOMED_INSET:
                    axins.fill(x, y, 
                               facecolor=COLOR_SWEPT, 
                               edgecolor=COLOR_SWEPT, 
                               alpha=0.25, 
                               zorder=2)
                    axins.plot(x, y, color=COLOR_SWEPT, linewidth=0.8, alpha=0.5, zorder=2)
        
        # 方法2: 样条插值生成平滑边界 (复杂轨迹可选)
        # 可以同时使用，选择效果更好的一种
        
        # 3. 绘制轨迹中心线 (增强视觉对比)
        ax.plot(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, 
                linewidth=1.0, linestyle='-', solid_capstyle='round',
                label='Path', zorder=4)
        ax.scatter(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
        if USE_ZOOMED_INSET:
            axins.plot(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, 
                       linewidth=1.0, linestyle='-', solid_capstyle='round',
                       zorder=4)
            axins.scatter(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
        # 添加中心线阴影增强立体感
        # ax.plot(traj_full[:, 0], traj_full[:, 1], color='white', linewidth=4.0, alpha=0.5, zorder=3)
        
        # 4. 可选: 稀疏显示几个关键车辆姿态 (增强运动感)
        # 选择几个关键点显示车辆轮廓
        rectangles = get_rect_points_vectorized(
            traj_full_clean, 
            width=1.8, 
            length=globalvar.vehicle_geometrics_.vehicle_length
        )
        n_points = len(traj_full_clean)
        key_indices = []
        if n_points > 2:
            step = max(1, n_points // 2)
            # step = 1
            key_indices = list(range(0, n_points, step))
            if n_points - 1 not in key_indices:
                key_indices.append(n_points - 1)
        else:
            key_indices = list(range(n_points))
            
        # 3. 绘制循环 (视觉优化版)
        for idx in key_indices:
            if idx < len(rectangles):
                rect = rectangles[idx]
                # 闭合矩形 (首尾相连)
                rect_closed = np.vstack([rect, rect[0]])
                
                # --- TMECH 风格配色逻辑 ---
                if idx == 0: 
                    # 起点：高亮实线框，强调起始状态
                    edge_color = 'black'       # 纯黑边缘
                    line_width = 1.8
                    alpha_fill = 0.0           # 内部不填充(透明)，只留框，或者填白色
                    z_order = 5                # 最上层
                    line_style = '-'
                    alpha_val = 1.0
                elif idx == n_points - 1:
                    # 终点：高亮实线框
                    edge_color = 'black'
                    line_width = 1.8
                    alpha_fill = 0.0
                    z_order = 5
                    line_style = '-'
                    alpha_val = 1.0
                else:
                    # 中间过程点：幽灵车 (Ghost Car) 效果
                    # 使用深色边缘，但线细一点，看起来像是在"流动的管道"里
                    edge_color = '#004d99'  # 深蓝
                    line_width = 0.8        # 变细
                    z_order = 3             # 略低于轨迹线
                    alpha_val = 0.7         # 线条稍微给一点透明度

                # 绘制边缘 (Edge)
                ax.plot(rect_closed[:, 0], rect_closed[:, 1], 
                       color=edge_color, 
                       linewidth=line_width, 
                       linestyle=line_style,
                       alpha=alpha_val, # 线条本身不透明
                       zorder=z_order)
                
                # (可选) 如果你想要车辆内部也有淡淡的颜色覆盖掉背景的网格
                if alpha_fill > 0:
                    ax.fill(rect_closed[:, 0], rect_closed[:, 1],
                            color='white', # 或者 COLOR_SWEPT
                            alpha=alpha_fill,
                            zorder=z_order-0.1) # 稍微在边框下面一点
        
        # 5. 绘制起点和终点 (增强视觉焦点)
        # 终点
        ax.scatter(target[0], target[1], color=COLOR_GOAL, s=200, 
                   marker='*', edgecolors='black', linewidth=1.0, 
                   zorder=5, label='Target')
        
        # 起点
        ax.scatter(traj_full[0, 0], traj_full[0, 1], color=COLOR_START, 
                   s=120, marker='o', edgecolors='black', linewidth=1.5, 
                   zorder=5, label='Start')
        
        # 6. 添加轨迹方向指示 (可选，增强运动感)
        # if len(traj_full) > 5:
        #     # 在轨迹中间添加一个箭头
        #     mid_idx = len(traj_full) // 2
        #     dx = 0.3 * np.cos(traj_full[mid_idx, 2])
        #     dy = 0.3 * np.sin(traj_full[mid_idx, 2])
        #     ax.arrow(traj_full[mid_idx, 0] - dx/2, traj_full[mid_idx, 1] - dy/2,
        #              dx, dy, head_width=0.2, head_length=0.3, 
        #              fc=COLOR_TRAJ, ec=COLOR_TRAJ, linewidth=1.5, zorder=4)
        
        # --- 高级图表美化 ---
        ax.axis('equal')
        
        # 设置坐标轴范围 (智能留边)
        margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
        margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
        ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
        ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
        
        # 简洁的坐标轴设置
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加细网格 (提高可读性)
        ax.grid(True, which='both', linestyle=':', linewidth=0.3, 
                color='gray', alpha=0.2)
        
        # 边框处理
        for spine in ax.spines.values():
            spine.set_visible(False)
            # spine.set_linewidth(0.5)
            # spine.set_color('gray')
            # spine.set_alpha(0.5)
        
        # 添加比例尺 (可选，增强学术性)
        scale_length = 5.0  # 10米比例尺
        scale_x = planning_scale_.xmin + 3.0
        scale_y = planning_scale_.ymin + 1.0
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
                'k-', linewidth=2, zorder=6)
        ax.text(scale_x + scale_length/2, scale_y - 0.3, f'{scale_length} m', 
                ha='center', va='top', fontsize=32)
        
        if USE_ZOOMED_INSET:
            x1, x2 = 14,20.2  # 关注区域的 X 轴范围
            y1, y2 = -2.5,1.5  # 关注区域的 Y 轴范围
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="#000000", lw=1.0)
        # 图例 (简洁版)
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(handles, labels, 
        #         loc='lower center',           # 中心对齐
        #         bbox_to_anchor=(0.5, 1.0),    # 放在绘图区域上方
        #         ncol=len(labels),             # 所有图例项排成一行
        #         frameon=True, 
        #         framealpha=0.9, 
        #         fancybox=False, 
        #         edgecolor='lightgray', 
        #         fontsize=18,
                
        #         borderpad=0.3,
        #         labelspacing=0.2,
        #         columnspacing=0.5,
        #         handletextpad=0.3,
        #         handlelength=1.2,
        #         handleheight=0.7
        #     )
        
        # 保存图片
        save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        while os.path.exists(save_file):
            i += 1
            save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        plt.tight_layout(pad=0.5)
        
        # plt.show()
        # 同时保存PDF和PNG和eps
        # plt.savefig(save_file.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', dpi=800)
        plt.savefig(save_file, format='png', dpi=800, bbox_inches='tight')
        plt.close()
        # print(f"Saved: {save_file}")
        
def visualize_data_batch_paper2(datas, trajectorys, save_path=None):
    """
    TMECH 投稿用：连续扫掠区域+中心线，高学术可视化质量
    """
    os.makedirs(save_path, exist_ok=True)
    # --- 高级配色方案 (优化对比度) ---
    COLOR_TRAJ = 'blue'       # 轨迹中心线
    COLOR_OBS = "#1E2C2C"        # 障碍物 (深石板灰)
    COLOR_GOAL = '#DC143C'       # 终点 (深红)
    COLOR_START = '#32CD32'      # 起点 (亮绿)
    COLOR_OBS_FILL = "#6E6C6C"   # 障碍物填充 (浅灰)
    COLOR_CAR = "#1DB0CA"
    USE_ZOOMED_INSET = False  # 是否使用放大镜效果展示细节
    # for i in range(1):
    for i in range(trajectorys.shape[0]):
        # if i < idxxx:
        #     continue
        data = {key: datas[key][i].cpu().numpy() if isinstance(datas[key][i], torch.Tensor) else datas[key][i] for key in datas}
        traj_full = trajectorys[i].cpu().detach().numpy() if isinstance(trajectorys[i], torch.Tensor) else trajectorys[i]
        obstacles_vertices = data['obstacles_vertices']
        target = data['target']
        
        traj_full_clean = traj_full
        traj_full = xy2xy_heading_numpy(traj_full)
        traj_full_clean = xy2xy_heading_numpy(traj_full_clean)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        if USE_ZOOMED_INSET:
            axins = zoomed_inset_axes(ax, zoom=2.0, loc='center', bbox_to_anchor=(0.28, 0.8), bbox_transform=ax.transAxes)
        
        # 1. 绘制障碍物 (优化视觉层次)
        polygons = obstacles_vertices.reshape(-1, 4, 2)
        for poly in polygons:
            ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                    linewidth=0.5, hatch='', alpha=1.0, zorder=1)
        # 3. 绘制轨迹中心线 (增强视觉对比)
        rectangles = get_rect_points_vectorized(
            traj_full_clean, 
            width=1.8, 
            length=globalvar.vehicle_geometrics_.vehicle_length
        )
        for rect in rectangles:  # 显示所有
            rect_closed = np.vstack([rect, rect[0]])  # 闭合矩形
            ax.plot(rect_closed[:, 0], rect_closed[:, 1], color=COLOR_CAR, linewidth=0.8, alpha=0.7, zorder=3)
            
        ax.plot(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, 
                linewidth=1.0, linestyle='-', solid_capstyle='round',
                label='Path', zorder=4)
        ax.scatter(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
        # traj_1 = traj_full[:25]
        # traj_2 = traj_full[24:27]
        # traj_3 = traj_full[26:]    
        # ax.plot(traj_1[:, 0], traj_1[:, 1], color=COLOR_TRAJ, 
        #         linewidth=1.0, linestyle='-', solid_capstyle='round',
        #         label='Path', zorder=4)
        # ax.scatter(traj_1[:, 0], traj_1[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
        
        # ax.plot(traj_3[:, 0], traj_3[:, 1], color=COLOR_TRAJ, 
        #         linewidth=1.0, linestyle='-', solid_capstyle='round',
        #         label='Path', zorder=4)
        # ax.scatter(traj_3[:, 0], traj_3[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
        
        # ax.plot(traj_2[:, 0], traj_2[:, 1], color='#8d2f25', 
        #         linewidth=1.0, linestyle='-', solid_capstyle='round',
        #         label='Path', zorder=4)
        # ax.scatter(traj_2[:, 0], traj_2[:, 1], color='#8d2f25', s=36, alpha=1.0, zorder=4)
        
        # if USE_ZOOMED_INSET:
        #     axins.plot(traj_1[:, 0], traj_1[:, 1], color=COLOR_TRAJ, 
        #             linewidth=1.0, linestyle='-', solid_capstyle='round',
        #             label='Path', zorder=4)
        #     axins.scatter(traj_1[:, 0], traj_1[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
            
        #     axins.plot(traj_3[:, 0], traj_3[:, 1], color=COLOR_TRAJ, 
        #             linewidth=1.0, linestyle='-', solid_capstyle='round',
        #             label='Path', zorder=4)
        #     axins.scatter(traj_3[:, 0], traj_3[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
            
        #     axins.plot(traj_2[:, 0], traj_2[:, 1], color='#8d2f25', 
        #             linewidth=1.0, linestyle='-', solid_capstyle='round',
        #             label='Path', zorder=4)
        #     axins.scatter(traj_2[:, 0], traj_2[:, 1], color='#8d2f25', s=36, alpha=1.0, zorder=4)
            
        #     rectangles = get_rect_points_vectorized(
        #         traj_full_clean, 
        #         width=1.8, 
        #         length=globalvar.vehicle_geometrics_.vehicle_length
        #     )
        #     for rect in rectangles:  # 显示所有
        #         rect_closed = np.vstack([rect, rect[0]])  # 闭合矩形
        #         axins.plot(rect_closed[:, 0], rect_closed[:, 1], color=COLOR_CAR, linewidth=0.8, alpha=0.7, zorder=3)    

        #     x1, x2, y1, y2 = 16, 24, 0.5, 5.2  # 你想看清的局部区域边界
        #     axins.set_xlim(x1, x2)
        #     axins.set_ylim(y1, y2)

        #     # 可选：如果你不想让放大图显示坐标轴刻度，可以加上下面两行
        #     # axins.set_xticks([])
        #     # axins.set_yticks([])

        #     # 5. 自动画出放大区域的方框和连接线
        #     # loc1=2, loc2=4 分别代表连接线连在主图方框和放大图的哪两个角（通常2和4，或者1和3，可以自己试试看哪个好看）
        #     # 示例：改成红色、加粗、虚线、带一点透明度的引导线
        #     mark_inset(
        #         ax, axins, 
        #         loc1=1, loc2=3, 
        #         fc="none",          # fc="none" 表示原图上的小方框内部不填充颜色
        #         ec="black",           # 颜色改为红色
        #         lw=1.5,             # 线条粗细改为 1.5
        #         ls="--",            # 线型改为虚线
        #         alpha=0.7           # 透明度设为 0.7
        #     )
        # 5. 绘制起点和终点 (增强视觉焦点)
        # 终点
        ax.scatter(target[0], target[1], color=COLOR_GOAL, s=200, 
                   marker='*', edgecolors='black', linewidth=1.0, 
                   zorder=5, label='Target')
        
        # 起点
        ax.scatter(traj_full[0, 0], traj_full[0, 1], color=COLOR_START, 
                   s=120, marker='o', edgecolors='black', linewidth=1.5, 
                   zorder=5, label='Start')
        
        ax.axis('equal')
        
        # 设置坐标轴范围 (智能留边)
        margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
        margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
        ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
        ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
        
        # 简洁的坐标轴设置
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加细网格 (提高可读性)
        ax.grid(True, which='both', linestyle=':', linewidth=0.3, 
                color='gray', alpha=0.2)
        
        # 边框处理
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # 保存图片
        save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        # while os.path.exists(save_file):
        #     i += 1
        #     save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        save_file_pdf = save_file.replace('.png', '.pdf')
        while os.path.exists(save_file_pdf):
            i += 1
            save_file_pdf = os.path.join(save_path, f"traj_swept_{i:03d}.pdf")
        plt.tight_layout(pad=0.5)
        
        # 同时保存PDF和PNG和eps
        # plt.savefig(save_file, format='png', dpi=800, bbox_inches='tight')
        plt.savefig(save_file_pdf, format='pdf', bbox_inches='tight', dpi=800)
        print(f"Saved: {save_file_pdf}")
        # plt.show()
        plt.close()
        
def visualize_single_data(datas, trajectorys, save_path=None, i = 0):
    """
    TMECH 投稿用：连续扫掠区域+中心线，高学术可视化质量
    """
    os.makedirs(save_path, exist_ok=True)
    # --- 高级配色方案 (优化对比度) ---
    COLOR_TRAJ = 'blue'       # 轨迹中心线
    COLOR_OBS = "#1E2C2C"        # 障碍物 (深石板灰)
    COLOR_GOAL = '#DC143C'       # 终点 (深红)
    COLOR_START = '#32CD32'      # 起点 (亮绿)
    COLOR_OBS_FILL = "#6E6C6C"   # 障碍物填充 (浅灰)
    COLOR_CAR = "#1DB0CA"
    
    data = {key: datas[key].cpu().numpy() if isinstance(datas[key], torch.Tensor) else datas[key] for key in datas}
    traj_full = trajectorys.cpu().detach().numpy() if isinstance(trajectorys, torch.Tensor) else trajectorys
    obstacles_vertices = data['obstacles_vertices']
    target = data['target']
    
    traj_full_clean = traj_full
    traj_full = xy2xy_heading_numpy(traj_full)
    traj_full_clean = xy2xy_heading_numpy(traj_full_clean)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # 1. 绘制障碍物 (优化视觉层次)
    polygons = obstacles_vertices.reshape(-1, 4, 2)
    for poly in polygons:
        ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                linewidth=0.5, hatch='', alpha=1.0, zorder=1)
    # 3. 绘制轨迹中心线 (增强视觉对比)
    ax.plot(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, 
            linewidth=1.0, linestyle='-', solid_capstyle='round',
            label='Path', zorder=4)
    ax.scatter(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
    
    rectangles = get_rect_points_vectorized(
        traj_full_clean, 
        width=1.8, 
        length=globalvar.vehicle_geometrics_.vehicle_length
    )
    for rect in rectangles:  # 显示所有
        rect_closed = np.vstack([rect, rect[0]])  # 闭合矩形
        ax.plot(rect_closed[:, 0], rect_closed[:, 1], color=COLOR_CAR, linewidth=0.8, alpha=0.7, zorder=3)
    # 5. 绘制起点和终点 (增强视觉焦点)
    # 终点
    ax.scatter(target[0], target[1], color=COLOR_GOAL, s=200, 
                marker='*', edgecolors='black', linewidth=1.0, 
                zorder=5, label='Target')
    
    # 起点
    ax.scatter(traj_full[0, 0], traj_full[0, 1], color=COLOR_START, 
                s=120, marker='o', edgecolors='black', linewidth=1.5, 
                zorder=5, label='Start')
    
    ax.axis('equal')
    
    # 设置坐标轴范围 (智能留边)
    margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
    margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
    ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
    ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
    
    # 简洁的坐标轴设置
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加细网格 (提高可读性)
    ax.grid(True, which='both', linestyle=':', linewidth=0.3, 
            color='gray', alpha=0.2)
    
    # 边框处理
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 保存图片
    save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
    save_file_pdf = save_file.replace('.png', '.pdf')
    while os.path.exists(save_file_pdf):
        i += 1
        save_file_pdf = os.path.join(save_path, f"traj_swept_{i:03d}.pdf")
    plt.tight_layout(pad=0.5)
    
    # 同时保存PDF和PNG和eps
    plt.savefig(save_file_pdf, format='pdf', bbox_inches='tight', dpi=800)
    # plt.savefig(save_file, format='png', dpi=800, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_file_pdf}")

def get_rect_points_vectorized(xy_heading, width=0.5, length=1.0):
    '''
    纯 PyTorch 版本，效率更高
    输入: xy_heading: (N, 3) 包含(x, y, heading)
    输出: rectangles: (N, 4, 2) 包含每个矩形的4个顶点坐标
    '''
    import torch
    
    # 确保是 PyTorch 张量
    if not isinstance(xy_heading, torch.Tensor):
        xy_heading = torch.tensor(xy_heading, dtype=torch.float32)
    
    N = xy_heading.shape[0]
    cos_h = torch.cos(xy_heading[:, 2])  # (N,)
    sin_h = torch.sin(xy_heading[:, 2])  # (N,)

    # 计算矩形四个顶点相对于中心点的偏移
    half_w = width / 2.0
    half_l = length / 2.0

    # 定义四个顶点的相对偏移量 (4, 2)
    offsets = torch.tensor([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w]
    ], dtype=xy_heading.dtype, device=xy_heading.device)

    # 向量化旋转计算
    rotated_offsets = torch.zeros((N, 4, 2), dtype=xy_heading.dtype, device=xy_heading.device)
    
    # 对每个顶点进行旋转
    for i in range(4):
        dx, dy = offsets[i]
        rotated_offsets[:, i, 0] = dx * cos_h - dy * sin_h
        rotated_offsets[:, i, 1] = dx * sin_h + dy * cos_h

    # 添加中心点坐标
    centers = xy_heading[:, :2].unsqueeze(1)  # (N, 1, 2)
    rectangles = centers + rotated_offsets  # (N, 4, 2)
    
    return rectangles

# def path_smoothness(path):
#     """
#     计算轨迹的平滑度指标, 根据曲率变化率计算
#     计算是否有小于最小转弯半径的点
#     path: numpy array of shape (N, 2)
#     """
#     path = np.array(path)
#     if len(path) < 3:
        
#         return 1.0, 1.0  # 太短的路径，无法计算曲率，返回默认值
#     # diff 计算曲率
#     dxs = np.diff(path[:, 0])
#     dys = np.diff(path[:, 1])
#     ddxs = np.diff(dxs)
#     ddys = np.diff(dys)
#     numerator = np.abs(dxs[:-1] * ddys - dys[:-1] * ddxs)
#     denominator = (dxs[:-1]**2 + dys[:-1]**2)**1.5 + 1e-3  # 防止除零
#     curvatures = numerator / denominator
    
#     # 计算曲率变化率
#     curvature_changes = np.abs(np.diff(curvatures))
#     smoothness = np.mean(curvature_changes)
    
#     min_turning_radius = vehicle_kinematics_.min_turning_radius
#     max_curvature = 1.0 / min_turning_radius
    
#     score_per_point = np.clip(max_curvature / curvatures, 0, 1)
#     score = np.mean(score_per_point)

#     return smoothness, score

def path_smoothness(path):
    """
    计算轨迹的平滑度指标, 根据曲率变化率计算
    计算是否有小于最小转弯半径的点
    path: numpy array of shape (N, 2)
    """
    path = np.array(path)
    if len(path) < 3:
        
        return 1.0, 1.0  # 太短的路径，无法计算曲率，返回默认值
    # 三点法计算曲率
    curvatures = []
    for i in range(1, len(path) - 1):
        p1 = path[i - 1]
        p2 = path[i]
        p3 = path[i + 1]

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a == 0 or b == 0 or c == 0:
            curvature = 0
        # 共线情况
        elif abs(a + b - c) < 1e-6 or abs(b + c - a) < 1e-6 or abs(c + a - b) < 1e-6:
            curvature = 0
        else:
            curvature = (np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) / (a * b * c)
        
        curvatures.append(curvature)
    curvatures = np.array(curvatures)
    # 计算曲率变化率
    curvature_changes = np.abs(np.diff(curvatures))
    smoothness = np.mean(curvature_changes)
    
    min_turning_radius = vehicle_kinematics_.min_turning_radius
    max_curvature = 1.0 / min_turning_radius
    
    radius = 1.0 / (curvatures + 1e-6)
    score_per_point = np.clip(radius / min_turning_radius, 0, 1)
    score = np.mean(score_per_point)

    return smoothness, score

def visualize_data_batch(datas, trajectorys, save_path=None):
    """
    可视化距离地图、障碍物顶点、初始点和终端点
    """
    os.makedirs(save_path, exist_ok=True)
    xy_heading = xy2xy_heading(trajectorys)
    for i in range(xy_heading.shape[0]):
        data = {key: datas[key][i].cpu().numpy() for key in datas}
        trajectory = xy_heading[i].cpu().detach().numpy() if isinstance(xy_heading, torch.Tensor) else xy_heading[i]
        # trajectory = trajectory[:-1, :]  # 去掉最后一个点
        obstacles_vertices = data['obstacles_vertices']
        target = data['target']
        plt.figure()
        
        # 绘制终点，用红色星号表示
        terminal_point = (target[0], target[1])
        plt.plot(terminal_point[0], terminal_point[1], 'r*', markersize=15, label='Terminal Point')  # 终端点
        # 绘制障碍物
        polygons = obstacles_vertices.reshape(-1, 4, 2) # 假设每个障碍物是一个四边形
        for polygon in polygons:
            plt.fill(polygon[:, 0], polygon[:, 1], 'k', alpha=0.5)

        # plt.plot(initial_point[0], initial_point[1], 'go', label='Initial Point')  # 初始点
        # plt.plot(terminal_point[0], terminal_point[1], 'ro', label='Terminal Point')  # 终端点
        
        # 绘制轨迹
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', label='Trajectory')
        # plt.scatter(trajectory[-1, 0], trajectory[-1, 1])
        # 绘制车辆矩形

        rectangles = get_rect_points_vectorized(trajectory, width=globalvar.vehicle_geometrics_.vehicle_width, length=globalvar.vehicle_geometrics_.vehicle_length)
        for rect in rectangles:
            rect = np.vstack([rect, rect[0]])  # 闭合矩形
            plt.plot(rect[:, 0], rect[:, 1], 'r-')
        index = i
        save_file = f"{save_path}/visualization_{index}.png"
        # while os.path.exists(save_file):
        #     index += 1
        #     save_file = f"{save_path}/visualization_{index}.png"
        plt.legend()
        plt.axis('equal')
        plt.xlim(planning_scale_.xmin, planning_scale_.xmax)
        plt.ylim(planning_scale_.ymin, planning_scale_.ymax)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Environment Visualization')
        if save_path:
            plt.savefig(save_file)
        plt.close()

def visualize_data_batch_2(datas, trajectorys_pred, trajectorys_final, save_path=None):
    """
    可视化距离地图、障碍物顶点、初始点和终端点
    """
    os.makedirs(save_path, exist_ok=True)
    xy_heading_pred = xy2xy_heading(trajectorys_pred)
    xy_heading_final = xy2xy_heading(trajectorys_final)
    for i in range(xy_heading_pred.shape[0]):
        data = {key: datas[key][i].cpu().numpy() for key in datas}
        trajectory_pred = xy_heading_pred[i].cpu().detach().numpy()
        trajectory_final = xy_heading_final[i].cpu().detach().numpy()
        # trajectory = trajectory[:-1, :]  # 去掉最后一个点
        obstacles_vertices = data['obstacles_vertices']
        target = data['target']
        plt.figure()
        
        # 绘制终点，用红色星号表示
        terminal_point = (target[0], target[1])
        plt.plot(terminal_point[0], terminal_point[1], 'r*', markersize=15, label='Terminal Point')  # 终端点
        # 绘制障碍物
        polygons = obstacles_vertices.reshape(-1, 4, 2) # 假设每个障碍物是一个四边形
        for polygon in polygons:
            plt.fill(polygon[:, 0], polygon[:, 1], 'k', alpha=0.5)

        # plt.plot(initial_point[0], initial_point[1], 'go', label='Initial Point')  # 初始点
        # plt.plot(terminal_point[0], terminal_point[1], 'ro', label='Terminal Point')  # 终端点
        
        # 绘制轨迹
        plt.plot(trajectory_pred[:, 0], trajectory_pred[:, 1], '-o', label='Trajectory (Predicted)', color='blue', alpha=0.5)
        plt.plot(trajectory_final[:, 0], trajectory_final[:, 1], '-o', label='Trajectory (Final)', color='orange', alpha=0.5)
        # plt.scatter(trajectory[-1, 0], trajectory[-1, 1])
        # 绘制车辆矩形

        # rectangles = get_rect_points_vectorized(trajectory, width=globalvar.vehicle_geometrics_.vehicle_width, length=globalvar.vehicle_geometrics_.vehicle_length)
        # for rect in rectangles:
        #     rect = np.vstack([rect, rect[0]])  # 闭合矩形
        #     plt.plot(rect[:, 0], rect[:, 1], 'r-')
        index = i
        save_file = f"{save_path}/visualization_{index}.png"
        # while os.path.exists(save_file):
        #     index += 1
        #     save_file = f"{save_path}/visualization_{index}.png"
        plt.legend()
        plt.axis('equal')
        plt.xlim(planning_scale_.xmin, planning_scale_.xmax)
        plt.ylim(planning_scale_.ymin, planning_scale_.ymax)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Environment Visualization')
        if save_path:
            plt.savefig(save_file)
        plt.close()
        
def VisualizeStaticResults(trajectory,obstacles_):
    # obstacles_ = globalvar.obstacles_ 
    nstep = len(trajectory.x)
    obstacles_ = np.array(obstacles_)
    obstacles_ = obstacles_.reshape(-1,4,2)
    ## plot obstacle
    if Nobs > 0:
        for j in range(0,Nobs):
            vertex_x = obstacles_[j, :, 0]
            vertex_y = obstacles_[j, :, 1]
            plt.fill(vertex_x,vertex_y,'k', alpha=0.5)
            # plt.hold(True)
    # plt.show()
    ## plot the planned trajectory
    plt.plot(trajectory.x,trajectory.y,'-o')
    # plt.hold(True)
    ## plot vehicle body
    for i in range(0,nstep):
        px = trajectory.x[i]
        py = trajectory.y[i]
        pth = trajectory.theta[i]
        V = CreateVehiclePolygon(px,py,pth)
        plt.plot(V.x,V.y,color='r')
        # plt.hold(True)

    ## plot start and terminal point
    # plt.plot(trajectory.x[0],trajectory.y[0],'o',color=(1,201 / 255,14 / 255),lw=1)
    # plt.plot(trajectory.x[nstep-1],trajectory.y[nstep-1],'p',color=(1,201 / 255,14 / 255),lw=1)
    plt.axis(np.array([planning_scale_.xmin,planning_scale_.xmax,planning_scale_.ymin,planning_scale_.ymax]))
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # plt.hold(True)

    plt.show()
    return


import warnings

def VisualizeDynamicResults(trajectory,obstacles_):
    warnings.simplefilter("ignore")
    plt.ion()
    planning_scale_ = globalvar.planning_scale_
    nstep = len(trajectory.x)
    plt.axis('equal')
    for i in range(nstep):
        plt.cla()
        ## plot obstacle
        if Nobs > 0:
            for j in range(0,Nobs):
                vertex_x = obstacles_[j, :, 0]
                vertex_y = obstacles_[j, :, 1]
                plt.fill(vertex_x,vertex_y,color=(0.7451,0.7451,0.7451))
        
        plt.axis(np.array([planning_scale_.xmin,planning_scale_.xmax,planning_scale_.ymin,planning_scale_.ymax]))
        plt.axis('equal')

        ## plot the planned trajectory
        plt.plot(trajectory.x,trajectory.y,'.-',markersize=2,lw = 1)
        ## plot vehicle body
        px = trajectory.x[i]
        py = trajectory.y[i]
        pth = trajectory.theta[i]
        V = CreateVehiclePolygon(px,py,pth)
        
        plt.plot(V.x,V.y,lw = 1)
        plt.pause(0.05)


    ## plot start and terminal point
    plt.plot(trajectory.x[0],trajectory.y[0],'o',lw = 1)
    plt.plot(trajectory.x[nstep-1],trajectory.y[nstep-1],'p',lw = 1)
    plt.axis(np.array([planning_scale_.xmin,planning_scale_.xmax,planning_scale_.ymin,planning_scale_.ymax]))
    plt.axis('equal')
    plt.xlabel('x (m)',fontsize=12)
    plt.ylabel('y (m)',fontsize=12)
    plt.ioff()
    plt.show()
    return

def inpolygon(x, y, xv, yv):
    n = len(xv)
    inside = False
    p1x, p1y = xv[0], yv[0]
    for i in range(1, n+1):
        p2x, p2y = xv[i % n], yv[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
# 检查线段相交
def check_segment_intersection(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    A, B = p1, p2
    C, D = p3, p4
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
# 检查多边形是否简单（无自相交）
def is_simple_polygon(poly):
    n = poly.shape[1]
    for i in range(n):
        for j in range(i+1, n):
            if check_segment_intersection(
                poly[:,i], poly[:,(i+1)%n],
                poly[:,j], poly[:,(j+1)%n]
            ):
                return False
    return True

def polygon_edges(vertices):
    ''''
    -vertices: 四边形顶点坐标，形状为 (n,4,2)
    '''
    # 将输入转成numpy数组
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)
    n = vertices.shape[0]
    edges = []
    for i in range(n):
        v = vertices[i]
        edge_set = []
        for j in range(4):
            x1, y1 = v[j]
            x2, y2 = v[(j + 1) % 4]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            edge_set.append((a, b, c))
        edges.append(edge_set)
    return edges # 将顶点转换为边的不等式参数 (a_i, b_i, c_i) 形状为 (n,4,3)

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

def h(x, y, polygons_edges, rho=10.0):
    '''
    -x: 点的x坐标
    -y: 点的y坐标
    -polygons_edges: 障碍物多边形的边列表，每个边是一个三元组 (a, b, c) 形状为 (m,4,3)
    '''
    all_distances = []
    for edge_set in polygons_edges:
        distances = []
        for a, b, c in edge_set:
            d = (a * x + b * y + c) / np.sqrt(a**2 + b**2)
            distances.append(d)
        all_distances.append(np.min(np.array(distances)))
    h1 = np.max(np.array(all_distances))
    return h1 + 1.5#+ 1.35 # 添加安全边距

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