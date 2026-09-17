import numpy as np
from typing import Callable, Dict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import globalvar
from utils.prob import xy2xy_heading
import os
try:
    from others.PlanHybridAStarPath import CreateVehiclePolygon
except ModuleNotFoundError:
    # The released training tree may omit the Hybrid A* source while retaining
    # only its visualization helper.  Keep plotting available in that layout.
    def CreateVehiclePolygon(x, y, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        half_width = globalvar.vehicle_geometrics_.vehicle_width * 0.5
        half_length = globalvar.vehicle_geometrics_.vehicle_length * 0.5
        polygon = globalvar.vclass()
        polygon.x = np.array([
            x + half_length * cos_theta - half_width * sin_theta,
            x + half_length * cos_theta + half_width * sin_theta,
            x - half_length * cos_theta + half_width * sin_theta,
            x - half_length * cos_theta - half_width * sin_theta,
            x + half_length * cos_theta - half_width * sin_theta,
        ])
        polygon.y = np.array([
            y + half_length * sin_theta + half_width * cos_theta,
            y + half_length * sin_theta - half_width * cos_theta,
            y - half_length * sin_theta - half_width * cos_theta,
            y - half_length * sin_theta + half_width * cos_theta,
            y + half_length * sin_theta + half_width * cos_theta,
        ])
        return polygon
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

# --- Global plotting style (preferably placed near the start of the module) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # Typeface preferred by IEEE
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['mathtext.fontset'] = 'stix' # Typeface for mathematical expressions


def path_clean(path, target):
    """
    Clean a path by removing duplicate points and points that are too close together.
    """
    # # Find the point closest to the goal
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
            # Delete points from min_index+1 through i-1
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
            if dist >= 0.5:  # Retain points separated by at least the threshold
                next_p = path[j]
                next_index = j
                break
        if next_p is not None:
            if np.linalg.norm(end_point[:2] - next_p[:2]) < 0.5:
                break
            cleaned_path.append(next_p)
        i = next_index if next_index > i else i + 1  # Prevent an infinite loop
    cleaned_path.append(path[-1])  # Ensure the goal point is included
    return np.array(cleaned_path)

def xy2xy_heading_numpy(xy):
    """
    Convert (x, y) points to (x, y, heading).
    - First point: forward difference.
    - Interior points: central difference using adjacent points.
    - Last point: backward difference.
    xy: shape (N, 2)
    Returns: shape (N, 3).
    """
    headings = np.zeros(len(xy))

    # First point: forward difference
    dx = xy[1, 0] - xy[0, 0]
    dy = xy[1, 1] - xy[0, 1]
    headings[0] = np.arctan2(dy, dx)

    # Interior points: central difference
    dx = xy[2:, 0] - xy[:-2, 0]
    dy = xy[2:, 1] - xy[:-2, 1]
    headings[1:-1] = np.arctan2(dy, dx)

    # Last point: backward difference
    dx = xy[-1, 0] - xy[-2, 0]
    dy = xy[-1, 1] - xy[-2, 1]
    headings[-1] = np.arctan2(dy, dx)

    return np.hstack([xy, headings.reshape(-1, 1)])

from shapely.geometry import Polygon
from shapely.ops import unary_union

def get_swept_path_as_polygon(traj_full, width, length, step=2):
    """
    Generate a continuous swept region using a geometric union.
    :param traj_full: Trajectory points [N, 3] as (x, y, theta).
    :param width: Vehicle width.
    :param length: Vehicle length.
    :param step: Sampling stride; process every few points for better performance.
    :return: A shapely.geometry.Polygon or MultiPolygon object.
    """
    if len(traj_full) < 2:
        return None

    polys = []
    # Half-length and half-width
    hl = length / 2.0
    hw = width / 2.0
    
    # A tiny buffer can avoid floating-point gaps, although union usually handles them
    # For extremely dense point sets, slight inflation can help merge the polygons
    
    # Traverse downsampled trajectory points
    for i in range(0, len(traj_full), step):
        x, y, theta = traj_full[i]
        c, s = np.cos(theta), np.sin(theta)
        
        # Compute four corners clockwise: FL, FR, RR, RL
        # Adjust for the coordinate system as needed; here x is forward and y is left
        corners = np.array([
            [x + hl*c - hw*s, y + hl*s + hw*c], # FL
            [x + hl*c + hw*s, y + hl*s - hw*c], # FR
            [x - hl*c + hw*s, y - hl*s - hw*c], # RR
            [x - hl*c - hw*s, y - hl*s + hw*c]  # RL
        ])
        polys.append(Polygon(corners))

    # Merge all rectangles into a single polygon
    swept_shape = unary_union(polys)
    
    return swept_shape

def generate_swept_contour_precise(traj_full, width, length):
    """
    Generate an accurate contour of the vehicle's swept volume.
    Connect corresponding corners of adjacent vehicle rectangles to form a continuous boundary.
    """
    if len(traj_full) < 2:
        return None
    
    # 1. Generate rectangle corners at each point
    rectangles = []
    for i in range(len(traj_full)):
        x, y, theta = traj_full[i]
        half_l = length / 2
        half_w = width / 2
        
        # Four vehicle corners in clockwise order
        corners = np.array([
            [x + half_l * np.cos(theta) - half_w * np.sin(theta),  # Front right
             y + half_l * np.sin(theta) + half_w * np.cos(theta)],
            [x + half_l * np.cos(theta) + half_w * np.sin(theta),  # Front left
             y + half_l * np.sin(theta) - half_w * np.cos(theta)],
            [x - half_l * np.cos(theta) + half_w * np.sin(theta),  # Rear left
             y - half_l * np.sin(theta) - half_w * np.cos(theta)],
            [x - half_l * np.cos(theta) - half_w * np.sin(theta),  # Rear right
             y - half_l * np.sin(theta) + half_w * np.cos(theta)]
        ])
        rectangles.append(corners)
    
    rectangles = np.array(rectangles)  # [n, 4, 2]
    
    # 2. Construct the outer boundary of the swept volume
    n = len(rectangles)
    
    # Left boundary: connect the front-left and rear-left corners of all rectangles
    left_boundary = []
    for i in range(n):
        left_boundary.append(rectangles[i, 1])  # Front-left corner
    for i in range(n-1, -1, -1):
        left_boundary.append(rectangles[i, 2])  # Rear-left corner in reverse order
    
    # Right boundary: connect the rear-right and front-right corners of all rectangles
    right_boundary = []
    for i in range(n):
        right_boundary.append(rectangles[i, 3])  # Rear-right corner
    for i in range(n-1, -1, -1):
        right_boundary.append(rectangles[i, 0])  # Front-right corner in reverse order
    
    # Combine the boundaries to form a polygon
    swept_contour = np.vstack([left_boundary, right_boundary, left_boundary[0:1]])
    
    return swept_contour

def densify_trajectory_smooth(traj, max_step=0.05, method='cubic'):
    """
    Densify the trajectory with higher-order spline interpolation for a smoother path.
    :param traj: [N, 3] numpy array (x, y, theta)
    :param max_step: Maximum interpolation step.
    :param method: 'cubic' for maximum smoothness or 'akima' to avoid overshoot.
    :return: Densified trajectory [M, 3].
    """
    if len(traj) < 3:
        # Too few points for spline interpolation; fall back to linear interpolation
        from scipy.interpolate import interp1d
        # ... The original logic can be used as a fallback ...
        return traj 

    x = traj[:, 0]
    y = traj[:, 1]
    yaw = traj[:, 2]

    # 1. Compute cumulative path distance (arc length)
    # This is the independent variable for interpolation
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dist[-1]

    # 2. Generate distances for the new sample points
    num_points = int(total_dist / max_step) + 1
    new_dists = np.linspace(0, total_dist, num_points)
    
    # 3. Select an interpolator
    # CubicSpline is smooth with continuous curvature but may overshoot at sharp turns
    # Akima1DInterpolator is slightly less smooth but more stable and avoids severe oscillations
    if method == 'cubic':
        Interpolator = CubicSpline
    else:
        Interpolator = Akima1DInterpolator

    # 4. Interpolate X and Y
    # bc_type='natural' sets endpoint curvature to zero and usually works best
    f_x = Interpolator(cum_dist, x)
    f_y = Interpolator(cum_dist, y)
    new_x = f_x(new_dists)
    new_y = f_y(new_dists)

    # 5. Interpolate yaw (critical step)
    # Unwrap first to prevent erroneous interpolation across the -pi/pi boundary
    yaw_unwrapped = np.unwrap(yaw)
    f_yaw = Interpolator(cum_dist, yaw_unwrapped)
    new_yaw = f_yaw(new_dists)
    
    # Optional: renormalize to the range [-pi, pi]
    # new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

    return np.column_stack((new_x, new_y, new_yaw))

def get_smooth_swept_path(traj_full, width, length, max_step=0.1, smooth_radius=0.2):
    """
    Generate a smooth swept region.
    :param max_step: Interpolation density; 0.05--0.1 is recommended.
    :param smooth_radius: Fillet radius; larger values produce rounder edges. Recommended: 0.1--0.3.
    """
    # 1. Densify the trajectory
    # dense_traj = densify_trajectory_smooth(traj_full, max_step=max_step)
    dense_traj = traj_full
    # 2. Generate densely sampled rectangles
    polys = []
    hl = length / 2.0
    hw = width / 2.0
    
    # Vectorize computations for better performance
    x = dense_traj[:, 0]
    y = dense_traj[:, 1]
    theta = dense_traj[:, 2]
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Precompute offsets for the four corners
    # FL, FR, RR, RL
    dx = np.array([hl, hl, -hl, -hl])
    dy = np.array([-hw, hw, hw, -hw]) # Check the left/right convention; this assumes the standard convention
    
    # Generate Polygon objects in a loop because Shapely construction is difficult to vectorize
    # Polygon object creation is the performance bottleneck here
    for i in range(len(dense_traj)):
        # Expand the rotation matrix manually
        # global_x = x + local_x * c - local_y * s
        # global_y = y + local_x * s + local_y * c
        corners_x = x[i] + dx * c[i] - dy * s[i]
        corners_y = y[i] + dx * s[i] + dy * c[i]
        
        polys.append(Polygon(np.column_stack((corners_x, corners_y))))

    # 3. Merge the polygons
    raw_union = unary_union(polys)
    
    # 4. Apply morphological smoothing (critical)
    # buffer(r) inflates and rounds the shape
    # buffer(-r) erodes it back to its original size
    # join_style=1 (ROUND) is essential for smoothing
    smoothed_shape = raw_union.buffer(smooth_radius, join_style=JOIN_STYLE.round) \
                              .buffer(-smooth_radius, join_style=JOIN_STYLE.round)
    
    return smoothed_shape

def animate_trajectory(traj_full, obstacles_vertices, target_pos, save_path="trajectory_animation.gif"):
    """
    Create a vehicle animation with a dynamically updated swept region.
    :param traj_full: Complete trajectory points [N, 3].
    :param obstacles_vertices: Array of obstacle vertices.
    :param target_pos: Goal coordinates [x, y].
    :param save_path: Output path for a .gif or .mp4 file.
    """
    
    # 1. Create the figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Color definitions consistent with the existing style
    COLOR_SWEPT = '#1f77b4'
    COLOR_TRAJ = '#FF4500'
    COLOR_OBS = '#2F4F4F'        # Obstacles (dark slate gray)
    COLOR_OBS_FILL = "#BEBEBE"
    
    # 2. Draw the static background: obstacles, start, and goal
    # Obstacles
    if obstacles_vertices is not None:
        polygons = obstacles_vertices.reshape(-1, 4, 2)
        for poly in polygons:
            ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                    linewidth=1.2, hatch='', alpha=1.0, zorder=1)
            
    # Goal
    ax.scatter(target_pos[0], target_pos[1], color='#DC143C', s=200, marker='*', 
               edgecolors='black', zorder=5, label='Target')
    # Start
    ax.scatter(traj_full[0, 0], traj_full[0, 1], color='#32CD32', s=120, marker='o', 
               edgecolors='black', zorder=5, label='Start')

    # Configure axes dynamically from the trajectory bounds
    margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
    margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
    ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
    ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
    ax.set_aspect('equal')
    ax.axis('off') # Hide axis ticks for a presentation-style animation

    # 3. Initialize dynamic artists
    # Swept region, initially empty
    swept_patch = None 
    
    # Trajectory line
    traj_line, = ax.plot([], [], color=COLOR_TRAJ, linewidth=2.5, zorder=4)
    
    # Current vehicle at its initial position
    current_car_patch = MplPolygon(np.zeros((4, 2)), closed=True, 
                                   fc='none', ec='black', lw=1.5, zorder=6)
    ax.add_patch(current_car_patch)

    # 4. Animation update function
    # Skip frames for faster rendering, for example frames=range(0, len, 2)
    def update(frame_idx):
        nonlocal swept_patch
        
        # Get data up to the current frame
        current_traj = traj_full[:frame_idx+1]
        current_pose = traj_full[frame_idx]
        
        # --- A. Update the trajectory line ---
        traj_line.set_data(current_traj[:, 0], current_traj[:, 1])
        
        # --- B. Update the current vehicle pose ---
        # Compute the vehicle rectangle for the current frame
        rects = get_rect_points_vectorized(
            np.array([current_pose]), 
            width=globalvar.vehicle_geometrics_.vehicle_width, 
            length=globalvar.vehicle_geometrics_.vehicle_length
        )
        current_car_patch.set_xy(rects[0])
        
        # --- C. Update the swept region, the most expensive step ---
        # Increase max_step or update only when frame_idx % 5 == 0 if rendering is too slow
        if frame_idx > 1:
            # Remove the previous frame's region
            if swept_patch:
                swept_patch.remove()
                
            # Compute the new cumulative region
            # Pass current_traj so only the traversed path is included
            poly = get_smooth_swept_path(
                current_traj, 
                width=globalvar.vehicle_geometrics_.vehicle_width, 
                length=globalvar.vehicle_geometrics_.vehicle_length,
                max_step=0.05,    # Slightly lower precision improves animation speed
                smooth_radius=0.15
            )
            
            # Convert the Shapely polygon to a Matplotlib patch
            if poly and not poly.is_empty:
                if poly.geom_type == 'Polygon':
                    xs, ys = poly.exterior.xy
                    swept_patch = MplPolygon(np.column_stack((xs, ys)), 
                                             fc=COLOR_SWEPT, ec='none', alpha=0.3, zorder=2)
                    ax.add_patch(swept_patch)
                elif poly.geom_type == 'MultiPolygon':
                    # MultiPolygon handling is more involved; use the largest part or merge them
                    # Animation trajectories are usually continuous, so MultiPolygon is rare
                    pass 

        return traj_line, current_car_patch, swept_patch

    # 5. Generate the animation
    # frames controls the frame count; subsample with range(0, len(traj_full), 2) if needed
    print("Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=range(0, len(traj_full), 1), 
                                  interval=1, blit=False, repeat=False)
    
    # 6. Save the animation
    if save_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=10)
        ani.save(save_path, writer=writer)
    elif save_path.endswith('.mp4'):
        # Requires ffmpeg
        writer = animation.FFMpegWriter(fps=20, extra_args=['-vcodec', 'libx264'])
        ani.save(save_path, writer=writer)
        
    print(f"Animation saved to: {save_path}")
    plt.close()

def visualize_data_batch_paper(datas, trajectorys, save_path=None):
    """
    Publication-quality TMECH visualization with a continuous swept region and centerline.
    """
    os.makedirs(save_path, exist_ok=True)
    USE_ZOOMED_INSET = False  # Whether to show details with a zoomed inset
    # --- High-contrast color palette ---
    COLOR_SWEPT = "#187dc5"      # Trajectory swept region; primary color with adjusted opacity
    COLOR_TRAJ = 'blue'       # Trajectory centerline
    COLOR_OBS = "#1E2C2C"        # Obstacles (dark slate gray)
    COLOR_GOAL = '#DC143C'       # Goal (dark red)
    COLOR_START = '#32CD32'      # Start (bright green)
    COLOR_OBS_FILL = "#6E6C6C"   # Obstacle fill (light gray)
    
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

        # Prepend the historical point (-1,0) and start point (0,0)
        # traj_full = np.vstack((np.array([[ -1.0, 0.0]]),np.array([[0.0, 0.0]]), traj_full))
        # traj_full_clean = np.vstack((np.array([[ -1.0, 0.0]]),np.array([[0.0, 0.0]]), traj_full_clean))
        # Convert to (x, y, heading)
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
        
        # 1. Draw obstacles with improved visual hierarchy
        polygons = obstacles_vertices.reshape(-1, 4, 2)
        for poly in polygons:
            ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                    linewidth=0.5, hatch='', alpha=1.0, zorder=1)
            if USE_ZOOMED_INSET:
                axins.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                           linewidth=0.5, hatch='', alpha=1.0, zorder=1)
        
        # 2. Generate a continuous swept region
        # Obtain vehicle outlines at all path points
        swept_poly = get_smooth_swept_path(
            traj_full_clean, 
            width=1.8, 
            length=globalvar.vehicle_geometrics_.vehicle_length,
            max_step=0.01,    # Dense enough to avoid large gaps at turns
            smooth_radius=2.0 # Remove jagged edges
        )

        if swept_poly is not None:
            # Shapely may return Polygon or MultiPolygon if the path is discontinuous
            if swept_poly.geom_type == 'Polygon':
                shapes = [swept_poly]
            elif swept_poly.geom_type == 'MultiPolygon':
                shapes = swept_poly.geoms
            else:
                shapes = []

            for shape in shapes:
                x, y = shape.exterior.xy
                # Rendering notes:
                # 1. Match edge and face colors to avoid internal hairlines
                # 2. Use lower alpha to emphasize the region
                # 3. Enable antialiasing for smoother edges
                ax.fill(x, y, 
                        facecolor=COLOR_SWEPT, 
                        edgecolor=COLOR_SWEPT, # Match edge and fill colors
                        alpha=0.25, 
                        zorder=2,
                        label='Swept Area' if shape == shapes[0] else None) # Label only the first shape

                # For a crisp dark outline, draw only the outer boundary
                ax.plot(x, y, color=COLOR_SWEPT, linewidth=0.8, alpha=0.5, zorder=2)
                
                if USE_ZOOMED_INSET:
                    axins.fill(x, y, 
                               facecolor=COLOR_SWEPT, 
                               edgecolor=COLOR_SWEPT, 
                               alpha=0.25, 
                               zorder=2)
                    axins.plot(x, y, color=COLOR_SWEPT, linewidth=0.8, alpha=0.5, zorder=2)
        
        # Method 2: spline interpolation for a smooth boundary, optional for complex trajectories
        # Both methods can be evaluated and the better result selected
        
        # 3. Draw the trajectory centerline with stronger visual contrast
        ax.plot(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, 
                linewidth=1.0, linestyle='-', solid_capstyle='round',
                label='Path', zorder=4)
        ax.scatter(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
        if USE_ZOOMED_INSET:
            axins.plot(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, 
                       linewidth=1.0, linestyle='-', solid_capstyle='round',
                       zorder=4)
            axins.scatter(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
        # Add a centerline shadow for visual depth
        # ax.plot(traj_full[:, 0], traj_full[:, 1], color='white', linewidth=4.0, alpha=0.5, zorder=3)
        
        # 4. Optionally show sparse key vehicle poses to convey motion
        # Select key points at which to display vehicle outlines
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
            
        # 3. Rendering loop with visual refinements
        for idx in key_indices:
            if idx < len(rectangles):
                rect = rectangles[idx]
                # Close the rectangle by connecting its endpoints
                rect_closed = np.vstack([rect, rect[0]])
                
                # --- TMECH-style color logic ---
                if idx == 0: 
                    # Start: highlighted solid outline emphasizing the initial state
                    edge_color = 'black'       # Solid black edge
                    line_width = 1.8
                    alpha_fill = 0.0           # Transparent interior; retain only the outline
                    z_order = 5                # Top layer
                    line_style = '-'
                    alpha_val = 1.0
                elif idx == n_points - 1:
                    # Goal: highlighted solid outline
                    edge_color = 'black'
                    line_width = 1.8
                    alpha_fill = 0.0
                    z_order = 5
                    line_style = '-'
                    alpha_val = 1.0
                else:
                    # Intermediate points: ghost-car effect
                    # Use a thin dark edge to suggest motion through a flowing corridor
                    edge_color = '#004d99'  # Dark blue
                    line_width = 0.8        # Thin line
                    z_order = 3             # Slightly below the trajectory line
                    alpha_val = 0.7         # Slightly transparent line

                # Draw the edge
                ax.plot(rect_closed[:, 0], rect_closed[:, 1], 
                       color=edge_color, 
                       linewidth=line_width, 
                       linestyle=line_style,
                       alpha=alpha_val, # Line opacity
                       zorder=z_order)
                
                # Optionally add a faint vehicle fill over the background grid
                if alpha_fill > 0:
                    ax.fill(rect_closed[:, 0], rect_closed[:, 1],
                            color='white', # Alternatively, use COLOR_SWEPT
                            alpha=alpha_fill,
                            zorder=z_order-0.1) # Place just below the outline
        
        # 5. Draw the start and goal as visual focal points
        # Goal
        ax.scatter(target[0], target[1], color=COLOR_GOAL, s=200, 
                   marker='*', edgecolors='black', linewidth=1.0, 
                   zorder=5, label='Target')
        
        # Start
        ax.scatter(traj_full[0, 0], traj_full[0, 1], color=COLOR_START, 
                   s=120, marker='o', edgecolors='black', linewidth=1.5, 
                   zorder=5, label='Start')
        
        # 6. Optionally indicate trajectory direction to convey motion
        # if len(traj_full) > 5:
        #     # Add an arrow at the middle of the trajectory
        #     mid_idx = len(traj_full) // 2
        #     dx = 0.3 * np.cos(traj_full[mid_idx, 2])
        #     dy = 0.3 * np.sin(traj_full[mid_idx, 2])
        #     ax.arrow(traj_full[mid_idx, 0] - dx/2, traj_full[mid_idx, 1] - dy/2,
        #              dx, dy, head_width=0.2, head_length=0.3, 
        #              fc=COLOR_TRAJ, ec=COLOR_TRAJ, linewidth=1.5, zorder=4)
        
        # --- Advanced plot styling ---
        ax.axis('equal')
        
        # Set axis limits with adaptive margins
        margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
        margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
        ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
        ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
        
        # Minimal axis styling
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a fine grid for readability
        ax.grid(True, which='both', linestyle=':', linewidth=0.3, 
                color='gray', alpha=0.2)
        
        # Style the plot frame
        for spine in ax.spines.values():
            spine.set_visible(False)
            # spine.set_linewidth(0.5)
            # spine.set_color('gray')
            # spine.set_alpha(0.5)
        
        # Optionally add a scale bar for publication clarity
        scale_length = 5.0  # Scale-bar length in meters
        scale_x = planning_scale_.xmin + 3.0
        scale_y = planning_scale_.ymin + 1.0
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
                'k-', linewidth=2, zorder=6)
        ax.text(scale_x + scale_length/2, scale_y - 0.3, f'{scale_length} m', 
                ha='center', va='top', fontsize=32)
        
        if USE_ZOOMED_INSET:
            x1, x2 = 14,20.2  # X-axis range of the region of interest
            y1, y2 = -2.5,1.5  # Y-axis range of the region of interest
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="#000000", lw=1.0)
        # Compact legend
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(handles, labels, 
        #         loc='lower center',           # Center alignment
        #         bbox_to_anchor=(0.5, 1.0),    # Position above the plotting area
        #         ncol=len(labels),             # Arrange all legend entries in one row
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
        
        # Save the figure
        save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        while os.path.exists(save_file):
            i += 1
            save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        plt.tight_layout(pad=0.5)
        
        # plt.show()
        # Save PDF, PNG, and EPS versions
        # plt.savefig(save_file.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', dpi=800)
        plt.savefig(save_file, format='png', dpi=800, bbox_inches='tight')
        plt.close()
        # print(f"Saved: {save_file}")
        
def visualize_data_batch_paper2(datas, trajectorys, save_path=None):
    """
    Publication-quality TMECH visualization with a continuous swept region and centerline.
    """
    os.makedirs(save_path, exist_ok=True)
    # --- High-contrast color palette ---
    COLOR_TRAJ = 'blue'       # Trajectory centerline
    COLOR_OBS = "#1E2C2C"        # Obstacles (dark slate gray)
    COLOR_GOAL = '#DC143C'       # Goal (dark red)
    COLOR_START = '#32CD32'      # Start (bright green)
    COLOR_OBS_FILL = "#6E6C6C"   # Obstacle fill (light gray)
    COLOR_CAR = "#1DB0CA"
    USE_ZOOMED_INSET = False  # Whether to show details with a zoomed inset
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
        
        # 1. Draw obstacles with improved visual hierarchy
        polygons = obstacles_vertices.reshape(-1, 4, 2)
        for poly in polygons:
            ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                    linewidth=0.5, hatch='', alpha=1.0, zorder=1)
        # 3. Draw the trajectory centerline with stronger visual contrast
        rectangles = get_rect_points_vectorized(
            traj_full_clean, 
            width=1.8, 
            length=globalvar.vehicle_geometrics_.vehicle_length
        )
        for rect in rectangles:  # Display all rectangles
            rect_closed = np.vstack([rect, rect[0]])  # Close the rectangle
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
        #     for rect in rectangles:  # Display all rectangles
        #         rect_closed = np.vstack([rect, rect[0]])  # Close the rectangle
        #         axins.plot(rect_closed[:, 0], rect_closed[:, 1], color=COLOR_CAR, linewidth=0.8, alpha=0.7, zorder=3)    

        #     x1, x2, y1, y2 = 16, 24, 0.5, 5.2  # Bounds of the local region to inspect
        #     axins.set_xlim(x1, x2)
        #     axins.set_ylim(y1, y2)

        #     # Optionally hide tick marks in the zoomed inset with the following two lines
        #     # axins.set_xticks([])
        #     # axins.set_yticks([])

        #     # 5. Draw the inset box and connector lines automatically
        #     # loc1 and loc2 select which corners connect the main and inset boxes
        #     # Example: use a thicker, dashed, slightly transparent guide line
        #     mark_inset(
        #         ax, axins, 
        #         loc1=1, loc2=3, 
        #         fc="none",          # Leave the inset rectangle unfilled
        #         ec="black",           # Connector color
        #         lw=1.5,             # Set line width to 1.5
        #         ls="--",            # Use a dashed line
        #         alpha=0.7           # Set opacity to 0.7
        #     )
        # 5. Draw the start and goal as visual focal points
        # Goal
        ax.scatter(target[0], target[1], color=COLOR_GOAL, s=200, 
                   marker='*', edgecolors='black', linewidth=1.0, 
                   zorder=5, label='Target')
        
        # Start
        ax.scatter(traj_full[0, 0], traj_full[0, 1], color=COLOR_START, 
                   s=120, marker='o', edgecolors='black', linewidth=1.5, 
                   zorder=5, label='Start')
        
        ax.axis('equal')
        
        # Set axis limits with adaptive margins
        margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
        margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
        ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
        ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
        
        # Minimal axis styling
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a fine grid for readability
        ax.grid(True, which='both', linestyle=':', linewidth=0.3, 
                color='gray', alpha=0.2)
        
        # Style the plot frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save the figure
        save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        # while os.path.exists(save_file):
        #     i += 1
        #     save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
        save_file_pdf = save_file.replace('.png', '.pdf')
        while os.path.exists(save_file_pdf):
            i += 1
            save_file_pdf = os.path.join(save_path, f"traj_swept_{i:03d}.pdf")
        plt.tight_layout(pad=0.5)
        
        # Save PDF, PNG, and EPS versions
        # plt.savefig(save_file, format='png', dpi=800, bbox_inches='tight')
        plt.savefig(save_file_pdf, format='pdf', bbox_inches='tight', dpi=800)
        print(f"Saved: {save_file_pdf}")
        # plt.show()
        plt.close()
        
def visualize_single_data(datas, trajectorys, save_path=None, i = 0):
    """
    Publication-quality TMECH visualization with a continuous swept region and centerline.
    """
    os.makedirs(save_path, exist_ok=True)
    # --- High-contrast color palette ---
    COLOR_TRAJ = 'blue'       # Trajectory centerline
    COLOR_OBS = "#1E2C2C"        # Obstacles (dark slate gray)
    COLOR_GOAL = '#DC143C'       # Goal (dark red)
    COLOR_START = '#32CD32'      # Start (bright green)
    COLOR_OBS_FILL = "#6E6C6C"   # Obstacle fill (light gray)
    COLOR_CAR = "#1DB0CA"
    
    data = {key: datas[key].cpu().numpy() if isinstance(datas[key], torch.Tensor) else datas[key] for key in datas}
    traj_full = trajectorys.cpu().detach().numpy() if isinstance(trajectorys, torch.Tensor) else trajectorys
    obstacles_vertices = data['obstacles_vertices']
    target = data['target']
    
    traj_full_clean = traj_full
    traj_full = xy2xy_heading_numpy(traj_full)
    traj_full_clean = xy2xy_heading_numpy(traj_full_clean)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # 1. Draw obstacles with improved visual hierarchy
    polygons = obstacles_vertices.reshape(-1, 4, 2)
    for poly in polygons:
        ax.fill(poly[:, 0], poly[:, 1], color=COLOR_OBS_FILL, edgecolor=COLOR_OBS, 
                linewidth=0.5, hatch='', alpha=1.0, zorder=1)
    # 3. Draw the trajectory centerline with stronger visual contrast
    ax.plot(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, 
            linewidth=1.0, linestyle='-', solid_capstyle='round',
            label='Path', zorder=4)
    ax.scatter(traj_full[:, 0], traj_full[:, 1], color=COLOR_TRAJ, s=36, alpha=1.0, zorder=4)
    
    rectangles = get_rect_points_vectorized(
        traj_full_clean, 
        width=1.8, 
        length=globalvar.vehicle_geometrics_.vehicle_length
    )
    for rect in rectangles:  # Display all rectangles
        rect_closed = np.vstack([rect, rect[0]])  # Close the rectangle
        ax.plot(rect_closed[:, 0], rect_closed[:, 1], color=COLOR_CAR, linewidth=0.8, alpha=0.7, zorder=3)
    # 5. Draw the start and goal as visual focal points
    # Goal
    ax.scatter(target[0], target[1], color=COLOR_GOAL, s=200, 
                marker='*', edgecolors='black', linewidth=1.0, 
                zorder=5, label='Target')
    
    # Start
    ax.scatter(traj_full[0, 0], traj_full[0, 1], color=COLOR_START, 
                s=120, marker='o', edgecolors='black', linewidth=1.5, 
                zorder=5, label='Start')
    
    ax.axis('equal')
    
    # Set axis limits with adaptive margins
    margin_x = (planning_scale_.xmax - planning_scale_.xmin) * 0
    margin_y = (planning_scale_.ymax - planning_scale_.ymin) * 0
    ax.set_xlim(planning_scale_.xmin - margin_x, planning_scale_.xmax + margin_x)
    ax.set_ylim(planning_scale_.ymin - margin_y, planning_scale_.ymax + margin_y)
    
    # Minimal axis styling
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a fine grid for readability
    ax.grid(True, which='both', linestyle=':', linewidth=0.3, 
            color='gray', alpha=0.2)
    
    # Style the plot frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save the figure
    save_file = os.path.join(save_path, f"traj_swept_{i:03d}.png")
    save_file_pdf = save_file.replace('.png', '.pdf')
    while os.path.exists(save_file_pdf):
        i += 1
        save_file_pdf = os.path.join(save_path, f"traj_swept_{i:03d}.pdf")
    plt.tight_layout(pad=0.5)
    
    # Save PDF, PNG, and EPS versions
    plt.savefig(save_file_pdf, format='pdf', bbox_inches='tight', dpi=800)
    # plt.savefig(save_file, format='png', dpi=800, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_file_pdf}")

def get_rect_points_vectorized(xy_heading, width=0.5, length=1.0):
    '''
    Pure PyTorch implementation for better performance.
    Input: xy_heading with shape (N, 3), containing (x, y, heading).
    Output: rectangles with shape (N, 4, 2), containing four vertices per rectangle.
    '''
    import torch
    
    # Ensure the input is a PyTorch tensor
    if not isinstance(xy_heading, torch.Tensor):
        xy_heading = torch.tensor(xy_heading, dtype=torch.float32)
    
    N = xy_heading.shape[0]
    cos_h = torch.cos(xy_heading[:, 2])  # (N,)
    sin_h = torch.sin(xy_heading[:, 2])  # (N,)

    # Compute offsets of the four rectangle vertices from the center
    half_w = width / 2.0
    half_l = length / 2.0

    # Define relative offsets of the four vertices (4, 2)
    offsets = torch.tensor([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w]
    ], dtype=xy_heading.dtype, device=xy_heading.device)

    # Vectorized rotation computation
    rotated_offsets = torch.zeros((N, 4, 2), dtype=xy_heading.dtype, device=xy_heading.device)
    
    # Rotate each vertex
    for i in range(4):
        dx, dy = offsets[i]
        rotated_offsets[:, i, 0] = dx * cos_h - dy * sin_h
        rotated_offsets[:, i, 1] = dx * sin_h + dy * cos_h

    # Add center coordinates
    centers = xy_heading[:, :2].unsqueeze(1)  # (N, 1, 2)
    rectangles = centers + rotated_offsets  # (N, 4, 2)
    
    return rectangles

# def path_smoothness(path):
#     """
#     Compute a trajectory smoothness metric from the rate of curvature change.
#     Check whether any point violates the minimum turning radius.
#     path: numpy array of shape (N, 2)
#     """
#     path = np.array(path)
#     if len(path) < 3:
        
#         return 1.0, 1.0  # Path is too short for curvature; return defaults
#     # Compute curvature using finite differences
#     dxs = np.diff(path[:, 0])
#     dys = np.diff(path[:, 1])
#     ddxs = np.diff(dxs)
#     ddys = np.diff(dys)
#     numerator = np.abs(dxs[:-1] * ddys - dys[:-1] * ddxs)
#     denominator = (dxs[:-1]**2 + dys[:-1]**2)**1.5 + 1e-3  # Avoid division by zero
#     curvatures = numerator / denominator
    
#     # Compute the rate of curvature change
#     curvature_changes = np.abs(np.diff(curvatures))
#     smoothness = np.mean(curvature_changes)
    
#     min_turning_radius = vehicle_kinematics_.min_turning_radius
#     max_curvature = 1.0 / min_turning_radius
    
#     score_per_point = np.clip(max_curvature / curvatures, 0, 1)
#     score = np.mean(score_per_point)

#     return smoothness, score

def path_smoothness(path):
    """
    Compute a trajectory smoothness metric from the rate of curvature change.
    Check whether any point violates the minimum turning radius.
    path: numpy array of shape (N, 2)
    """
    path = np.array(path)
    if len(path) < 3:
        
        return 1.0, 1.0  # Path is too short for curvature; return defaults
    # Compute curvature using the three-point method
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
        # Collinear case
        elif abs(a + b - c) < 1e-6 or abs(b + c - a) < 1e-6 or abs(c + a - b) < 1e-6:
            curvature = 0
        else:
            curvature = (np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) / (a * b * c)
        
        curvatures.append(curvature)
    curvatures = np.array(curvatures)
    # Compute the rate of curvature change
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
    Visualize the distance map, obstacle vertices, start point, and goal point.
    """
    os.makedirs(save_path, exist_ok=True)
    xy_heading = xy2xy_heading(trajectorys)
    for i in range(xy_heading.shape[0]):
        data = {key: datas[key][i].cpu().numpy() for key in datas}
        trajectory = xy_heading[i].cpu().detach().numpy() if isinstance(xy_heading, torch.Tensor) else xy_heading[i]
        # trajectory = trajectory[:-1, :]  # Remove the last point
        obstacles_vertices = data['obstacles_vertices']
        target = data['target']
        plt.figure()
        
        # Draw the goal as a red star
        terminal_point = (target[0], target[1])
        plt.plot(terminal_point[0], terminal_point[1], 'r*', markersize=15, label='Terminal Point')  # Goal point
        # Draw obstacles
        polygons = obstacles_vertices.reshape(-1, 4, 2) # Assume every obstacle is a quadrilateral
        for polygon in polygons:
            plt.fill(polygon[:, 0], polygon[:, 1], 'k', alpha=0.5)

        # plt.plot(initial_point[0], initial_point[1], 'go', label='Initial Point')  # Start point
        # plt.plot(terminal_point[0], terminal_point[1], 'ro', label='Terminal Point')  # Goal point
        
        # Draw the trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', label='Trajectory')
        # plt.scatter(trajectory[-1, 0], trajectory[-1, 1])
        # Draw vehicle rectangles

        rectangles = get_rect_points_vectorized(trajectory, width=globalvar.vehicle_geometrics_.vehicle_width, length=globalvar.vehicle_geometrics_.vehicle_length)
        for rect in rectangles:
            rect = np.vstack([rect, rect[0]])  # Close the rectangle
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
    Visualize the distance map, obstacle vertices, start point, and goal point.
    """
    os.makedirs(save_path, exist_ok=True)
    xy_heading_pred = xy2xy_heading(trajectorys_pred)
    xy_heading_final = xy2xy_heading(trajectorys_final)
    for i in range(xy_heading_pred.shape[0]):
        data = {key: datas[key][i].cpu().numpy() for key in datas}
        trajectory_pred = xy_heading_pred[i].cpu().detach().numpy()
        trajectory_final = xy_heading_final[i].cpu().detach().numpy()
        # trajectory = trajectory[:-1, :]  # Remove the last point
        obstacles_vertices = data['obstacles_vertices']
        target = data['target']
        plt.figure()
        
        # Draw the goal as a red star
        terminal_point = (target[0], target[1])
        plt.plot(terminal_point[0], terminal_point[1], 'r*', markersize=15, label='Terminal Point')  # Goal point
        # Draw obstacles
        polygons = obstacles_vertices.reshape(-1, 4, 2) # Assume every obstacle is a quadrilateral
        for polygon in polygons:
            plt.fill(polygon[:, 0], polygon[:, 1], 'k', alpha=0.5)

        # plt.plot(initial_point[0], initial_point[1], 'go', label='Initial Point')  # Start point
        # plt.plot(terminal_point[0], terminal_point[1], 'ro', label='Terminal Point')  # Goal point
        
        # Draw the trajectories
        plt.plot(trajectory_pred[:, 0], trajectory_pred[:, 1], '-o', label='Trajectory (Predicted)', color='blue', alpha=0.5)
        plt.plot(trajectory_final[:, 0], trajectory_final[:, 1], '-o', label='Trajectory (Final)', color='orange', alpha=0.5)
        # plt.scatter(trajectory[-1, 0], trajectory[-1, 1])
        # Draw vehicle rectangles

        # rectangles = get_rect_points_vectorized(trajectory, width=globalvar.vehicle_geometrics_.vehicle_width, length=globalvar.vehicle_geometrics_.vehicle_length)
        # for rect in rectangles:
        #     rect = np.vstack([rect, rect[0]])  # Close the rectangle
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
# Check segment intersection
def check_segment_intersection(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    A, B = p1, p2
    C, D = p3, p4
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
# Check whether a polygon is simple, with no self-intersection
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
    -vertices: Quadrilateral vertex coordinates with shape (n, 4, 2).
    '''
    # Convert the input to a NumPy array
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
    return edges # Convert vertices to edge-inequality coefficients (a_i, b_i, c_i), shape (n, 4, 3)

def check_polygon_intersection(poly1, poly2):
    """
    Check whether two quadrilaterals intersect.
    Args:
        poly1, poly2: 2x4 NumPy arrays containing quadrilateral vertex coordinates,
                     formatted as [[x1,x2,x3,x4], [y1,y2,y3,y4]].
    Returns:
        bool: True if the quadrilaterals intersect; otherwise False.
    """
    # Validate input shapes
    if poly1.shape == (4,2):
        poly1 = poly1.T
    if poly2.shape == (4,2):
        poly2 = poly2.T
    
    if poly1.shape != (2,4) or poly2.shape != (2,4):
        raise ValueError("Input must be a 2x4 NumPy array")
    
    # Check edge intersections
    if check_edges_intersection(poly1, poly2):
        return True
    
    # Check containment
    if check_containment(poly1, poly2):
        return True
    
    return False

def h(x, y, polygons_edges, rho=10.0):
    '''
    -x: Point x-coordinate.
    -y: Point y-coordinate.
    -polygons_edges: Obstacle-polygon edges, each represented by coefficients
                     (a, b, c), with shape (m, 4, 3).
    '''
    all_distances = []
    for edge_set in polygons_edges:
        distances = []
        for a, b, c in edge_set:
            d = (a * x + b * y + c) / np.sqrt(a**2 + b**2)
            distances.append(d)
        all_distances.append(np.min(np.array(distances)))
    h1 = np.max(np.array(all_distances))
    return h1 + 1.5#+ 1.35 # Add a safety margin

def check_edges_intersection(poly1, poly2):
    """Check whether the edges of two quadrilaterals intersect."""
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
    """Check whether two line segments intersect."""
    # Ensure all points are two-dimensional coordinates
    a1 = np.asarray(a1).flatten()[:2]
    a2 = np.asarray(a2).flatten()[:2]
    b1 = np.asarray(b1).flatten()[:2]
    b2 = np.asarray(b2).flatten()[:2]
    
    # Use cross products to test for segment intersection
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    # Check the general intersection case
    case1 = ccw(a1, b1, b2) != ccw(a2, b1, b2)
    case2 = ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    if case1 and case2:
        return True
    
    # Check coincident endpoints
    if (np.array_equal(a1, b1) or np.array_equal(a1, b2) or 
        np.array_equal(a2, b1) or np.array_equal(a2, b2)):
        return True
    
    # Check collinear overlap
    if is_point_on_segment(a1, b1, b2) or is_point_on_segment(a2, b1, b2):
        return True
    if is_point_on_segment(b1, a1, a2) or is_point_on_segment(b2, a1, a2):
        return True
    
    return False

def is_point_on_segment(p, a, b):
    """Check whether point p lies on segment ab."""
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
    """Check whether either quadrilateral is fully contained in the other."""
    # Check whether all vertices of poly1 lie inside poly2
    if all(point_in_polygon(poly1[:,i], poly2) for i in range(4)):
        return True
    
    # Check whether all vertices of poly2 lie inside poly1
    if all(point_in_polygon(poly2[:,i], poly1) for i in range(4)):
        return True
    
    return False

def point_in_polygon(point, polygon):
    """Use ray casting to determine whether a point lies inside a quadrilateral."""
    x, y = point
    n = 4  # A quadrilateral has four vertices
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
