import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import globalvar
import math
from torch.nn.functional import grid_sample
def LSE_max(a, dim, rho=10.0):
    """
    Compute the smooth maximum of tensor a along the specified dimension.
    LSE_max(a) = (1/rho) * log(sum(exp(rho * a)))
    
    Args:
        a: Input tensor.
        dim: Dimension along which to reduce.
        rho: Smoothing parameter; larger values more closely approximate max.
        
    Returns:
        smooth_max: Tensor containing the smooth maximum values.
    """
    return (1.0 / rho) * torch.logsumexp(rho * a, dim=dim)

def LSE_min(a, dim, rho=10.0):
    """
    Compute the smooth minimum of tensor a along the specified dimension.
    LSE_min(a) = -(1/rho) * log(sum(exp(-rho * a)))
    
    Args:
        a: Input tensor.
        dim: Dimension along which to reduce.
        rho: Smoothing parameter; larger values more closely approximate min.
        
    Returns:
        smooth_min: Tensor containing the smooth minimum values.
    """
    return -(1.0 / rho) * torch.logsumexp(-rho * a, dim=dim)

def vertices_to_edges(polygons):
    # vertices shape: (batch_size, K, 4, 2)
    # Get vertex pairs (for quadrilateral)
    v1 = polygons
    v2 = torch.roll(polygons, shifts=-1, dims=2)  # Roll along the vertex dimension
    
    # Compute edge coefficients
    a = v2[:, :, :, 1] - v1[:, :, :, 1]  # y2 - y1
    b = v1[:, :, :, 0] - v2[:, :, :, 0]  # x1 - x2
    c = v2[:, :, :, 0] * v1[:, :, :, 1] - v1[:, :, :, 0] * v2[:, :, :, 1]  # x2*y1 - x1*y2
    
    # Stack coefficients along last dimension
    edges = torch.stack([a, b, c], dim=-1)  # (batch_size, K, 4, 3)
    return edges

def h(x, y, polygons, rho):
    # x, y: (batch_size, N) - N coordinates per batch
    # polygons: tensor of shape (batch_size, K, 4, 2) where K is number of quadrilaterals per batch
    
    batch_size, N = x.shape
    K = polygons.shape[1]
    
    # Convert vertices to edges
    edges = vertices_to_edges(polygons)  # (batch_size, K, 4, 3)
    
    # Expand dimensions for broadcasting
    # x and y will be (batch_size, N, 1, 1) to match (batch_size, K, 4) edges
    x_exp = x.view(batch_size, N, 1, 1)
    y_exp = y.view(batch_size, N, 1, 1)
    
    # Get edge coefficients
    a = edges[:, :, :, 0]  # (batch_size, K, 4)
    b = edges[:, :, :, 1]  # (batch_size, K, 4)
    c = edges[:, :, :, 2]  # (batch_size, K, 4)
    
    # Expand edge coefficients to match coordinates
    a_exp = a.unsqueeze(1)  # (batch_size, 1, K, 4)
    b_exp = b.unsqueeze(1)  # (batch_size, 1, K, 4)
    c_exp = c.unsqueeze(1)  # (batch_size, 1, K, 4)
    
    # Compute distances for all edges and all points
    numerator = a_exp * x_exp + b_exp * y_exp + c_exp  # (batch_size, N, K, 4)
    denominator = torch.sqrt(a_exp**2 + b_exp**2)  # (batch_size, N, K, 4)
    distances = numerator / denominator  # (batch_size, N, K, 4)
    
    # Determine the minimum distance from each point to the edges of each quadrilateral.
    quadrilateral_distances = LSE_min(distances, dim=-1)  # (batch_size, N, K)
    
    # For each point, find maximum distance across quadrilaterals
    point_results = LSE_max(quadrilateral_distances, dim=-1)  # (batch_size, N)
    point_results = point_results + globalvar.vehicle_geometrics_.Safety_margin  # Safety margin

    # result = F.relu(point_results)  # Only keep positive distances (B, N)
    # result = result.sum(dim=1, keepdim=True)  # (batch_size, 1)
    # result = LSE_max(point_results, dim=-1).unsqueeze(1)  # (batch_size, 1)
    return point_results # (batch_size, N)

def xy2xy_heading(xy):
    '''
    Input: xy with shape (B, N, 2).
    Output: xy_heading with shape (B, N+1, 3), including start point (0,0).
    '''
    if not isinstance(xy, torch.Tensor):
        xy = torch.from_numpy(xy)

    device = xy.device
    B, N, _ = xy.shape

    # Prepend the start point (0,0)
    init_point = torch.zeros((B, 1, 2), device=device)
    traj_points = torch.cat([init_point, xy], dim=1)  # (B, N+1, 2)

    # Central differences for points 0 through N-1
    delta_mid = traj_points[:, 2:, :] - traj_points[:, :-2, :]  # (B, N-1, 2)
    h_mid = torch.atan2(delta_mid[..., 1], delta_mid[..., 0])   # (B, N-1)

    # End point: backward difference
    delta_end = traj_points[:, -1, :] - traj_points[:, -2, :]
    h_end = torch.atan2(delta_end[:, 1], delta_end[:, 0]).unsqueeze(1)  # (B, 1)

    # Start point (0,0): forward difference toward the first predicted point
    delta_start = traj_points[:, 1, :] - traj_points[:, 0, :]
    h_start = torch.atan2(delta_start[:, 1], delta_start[:, 0]).unsqueeze(1)  # (B, 1)

    headings = torch.cat([h_start, h_mid, h_end], dim=1)  # (B, N+1)

    xy_heading = torch.cat([traj_points, headings.unsqueeze(2)], dim=2)  # (B, N+1, 3)
    return xy_heading

def get_safe_circle_centers(xy_heading):
    '''
    Input: xy_heading with shape (B, N, 3), containing (x, y, heading).
    Output: circle_centers with shape (B, N, 3), flattened to (B, N*3, 2).
    '''
    B, N, _ = xy_heading.shape
    delta_l = globalvar.vehicle_geometrics_.vehicle_length / 3.0
    
    # Extract base data to avoid repeated slicing
    x = xy_heading[:, :, 0]
    y = xy_heading[:, :, 1]
    heading = xy_heading[:, :, 2]
    
    # Precompute offsets
    # These operations create new tensors, as required by vmap
    offset_x = delta_l * torch.cos(heading)
    offset_y = delta_l * torch.sin(heading)

    # --- Compute the three circle centers using out-of-place operations ---
    center_mid = xy_heading[:, :, :2] 
    center_front = torch.stack([x + offset_x, y + offset_y], dim=-1)
    center_rear = torch.stack([x - offset_x, y - offset_y], dim=-1)

    # --- Stack the results ---
    # Stack along dim=2 to obtain shape (B, N, 3, 2)
    # Order: [middle, front, rear]; ordering does not affect collision detection
    circle_centers = torch.stack([center_mid, center_front, center_rear], dim=2)

    # Flatten the output
    return circle_centers.view(B, N*3, 2)


def compute_kappa_menger(xy, dist_threshold=0.2):
    '''
    Fully differentiable Menger circumcircle curvature.
    xy: Shape (B, N+1, 2), including start point (0,0).
    dist_threshold: Point-spacing threshold below which curvature approaches zero.
    return: kappa with shape (B, N), excluding the start point.
    '''
    eps = 1e-6

    P0 = xy[:, :-2, :]   # (B, N-1, 2)
    P1 = xy[:, 1:-1, :]  # (B, N-1, 2)
    P2 = xy[:, 2:,  :]   # (B, N-1, 2)

    a = ((P1 - P0).pow(2).sum(dim=-1) + eps).sqrt()  # (B, N-1)
    b = ((P2 - P1).pow(2).sum(dim=-1) + eps).sqrt()
    c = ((P2 - P0).pow(2).sum(dim=-1) + eps).sqrt()

    cross = (P1[..., 0] - P0[..., 0]) * (P2[..., 1] - P0[..., 1]) \
          - (P1[..., 1] - P0[..., 1]) * (P2[..., 0] - P0[..., 0])

    kappa_mid = 2.0 * cross.abs() / (a * b * c + eps)  # (B, N-1)

    # Use the shortest-edge proxy to determine whether points are sufficiently separated
    min_dist = (a + b) / 2.0

    # Soft tanh mask: approaches 0 far below the threshold and 1 far above it
    # tanh(x) is 0 at x=0 and nearly 1 at x=2--3, hence x = dist/threshold * 3
    weight = torch.tanh(min_dist / dist_threshold * 3.0)  # (B, N-1)

    kappa_mid = kappa_mid * weight

    kappa_full = torch.cat([kappa_mid[:, :1],
                            kappa_mid,
                            kappa_mid[:, -1:]], dim=1)  # (B, N+1)
    return kappa_full[:, 1:]  # (B, N)


def soft_constraints(xy_heading, obs_constraints_weight, obstacles_vertices):
    '''
    xy_heading(N+1)
    Constraint 1: The displacement at each step must not be too large.
    Constraint 2: The heading change at each step must not be too large.
    '''
    #d
    distances = torch.norm(xy_heading[:, 1:, :2] - xy_heading[:, :-1, :2], dim=2)  # (B, N)
    max_distance = 0.9 # Maximum displacement
    distance_violations = F.relu(distances - max_distance)  # (B, N)
    # Increase the weight of distance violations. For APF supervision signals, the weight of distance violations needs to be somewhat larger.
    distance_violations = (distance_violations ** 2 * 100.0)
    distance_violations = distance_violations.sum(dim=1, keepdim=True)  # (B, 1)
    
    turning_kappa = compute_kappa_menger(xy_heading[:, :, :2])  # (B, N-2)
    min_turning_radius = globalvar.vehicle_kinematics_.min_turning_radius  # Minimum turning radius
    max_turning_kappa = 1.0 / min_turning_radius
    turning_violations = F.relu(turning_kappa - max_turning_kappa)  # (B, N-2)
    # turning_violations = turning_violations.mean(dim=1, keepdim=True)*50*2  # (B, 1)
    turning_violations = LSE_max(turning_violations, dim=1, rho=20) * 20
    
    # obs
    xy_heading = xy_heading[:, 1:, :]  # Remove the start point (batch_size, N, 3)
    safe_centers = get_safe_circle_centers(xy_heading)  # (batch_size, N*3, 2)
    point_x = safe_centers[:, :, 0]  # (batch_size, N*3)
    point_y = safe_centers[:, :, 1]  # (batch_size, N*3)
    safety_distances = h(point_x, point_y, obstacles_vertices, rho=20.0)  # (batch_size, N*3)
    safety_distances = F.relu(safety_distances)  # Retain only positive values (B, N*3)
    # safety_distances = safety_distances.view(safety_distances.shape[0], -1, 3) # (B, N, 3)
    # safety_distances = LSE_max(safety_distances, dim=2)  # (B, N)
    safety_distances = torch.mean(safety_distances, dim=1, keepdim=True) # (B, 1)
    # trace_loss = torch.cat([distance_violations, turning_violations], dim=1)  # (B, 2)
    # soft_loss =trace_loss.mean() + safety_distances.mean() * 200
    soft_loss = distance_violations.mean() +turning_violations.mean() + safety_distances.mean() * obs_constraints_weight #distance_violations.mean() + 
    return soft_loss

def _create_objective_function(stage = 2):
    # stage 1 only update the slack
    def objective_function(data, y):
        y = y.view(y.shape[0], -1, 7)  # (batch_size, N, 2+5)
        xy = y[:, :, :2].detach() if stage == 1 else y[:, :, :2]
        s = y[:, :, 2:5]  # (batch_size, N, 3) slack variables
        s_2 = s ** 2
        s_h = y[:, :, 5]  # (batch_size, N) smoothness slack variables
        s_h_2 = s_h ** 2
        s_d = y[:, :, 6]  # (batch_size, N) distance slack variables
        s_d_2 = s_d ** 2

        xy_heading = xy2xy_heading(xy)  # (batch_size, N+1, 3) include (0,0)

        distances = torch.norm(xy_heading[:, 1:, :2] - xy_heading[:, 0:-1, :2], dim=2)  # (B, N)
        max_distance = 1.0
        distance_violations = distances - max_distance  # (B, N)
        residuals_distance = distance_violations + s_d_2  # (B, N)

        kappas = compute_kappa_menger(xy_heading[:, :, :2])  # (B, N)
        kappa_max = 1.0 / globalvar.vehicle_kinematics_.min_turning_radius  # Maximum curvature
        safety_kappas = kappas - kappa_max  # (B, N-1)
        residuals_kappa = safety_kappas + s_h_2  # (B, N-1)
        
        xy_heading = xy_heading[:, 1:, :]  # Remove the start point (batch_size, N, 3)
        safe_centers = get_safe_circle_centers(xy_heading)  # (batch_size, N*3, 2)
        point_x = safe_centers[:, :, 0]  # (batch_size, N*3)
        point_y = safe_centers[:, :, 1]  # (batch_size, N*3)
        obstacles_vertices = data['obstacles_vertices']  # (batch_size, K, 4, 2)
        safety_distances = h(point_x, point_y, obstacles_vertices, rho=20.0)  # (batch_size, N*3)
        safety_distances = safety_distances.view(y.shape[0], -1, 3)  # (batch_size, N, 3)
        residuals_safety = safety_distances + s_2 # (batch_size, N, 3)
        # Engineering note:
        # The paper formulates 3T separate collision constraints (one per body circle),
        # with a corresponding slack variable for each (total N_C = 200).
        # Here we aggregate the 3 per-waypoint circle residuals via LSE before 
        # stacking into the full residual vector, reducing the effective Jacobian 
        # from (200 x 280) to (120 x 280) for computational efficiency.
        # This is equivalent to the paper formulation when rho is sufficiently large.
        residuals_safety = LSE_max(residuals_safety, dim=2, rho=10)  # (batch_size, N)

        residuals = torch.cat([residuals_safety, residuals_kappa, residuals_distance], dim=1)  # (batch_size, 5N)
        
        return residuals
    
    return objective_function

def create_enforce_inequality_constraints():
    def constraints_fn(data, p):
        # p: (B, 2N), only trajectory output, no slack variables
        B = p.shape[0]
        xy = p.view(B, -1, 2)

        xy_heading = xy2xy_heading(xy)  # (B, N+1, 3), include start point

        # distance constraint: ||p_{t+1} - p_t|| - d_max <= 0
        distances = torch.norm(
            xy_heading[:, 1:, :2] - xy_heading[:, :-1, :2],
            dim=2
        )
        distance_violations = distances - 1.0  # (B, N)

        # curvature constraint: kappa - kappa_max <= 0
        kappas = compute_kappa_menger(xy_heading[:, :, :2])
        kappa_max = 1.0 / globalvar.vehicle_kinematics_.min_turning_radius
        kappa_violations = kappas - kappa_max  # (B, N) or (B, N-1), depending on your function

        # collision constraint
        xy_heading_no_start = xy_heading[:, 1:, :]
        safe_centers = get_safe_circle_centers(xy_heading_no_start)  # (B, N*3, 2)

        point_x = safe_centers[:, :, 0]
        point_y = safe_centers[:, :, 1]
        obstacles_vertices = data["obstacles_vertices"]

        safety_distances = h(
            point_x,
            point_y,
            obstacles_vertices,
            rho=20.0
        )  # (B, N*3)
        safety_distances = safety_distances.view(B, -1, 3)  # (B, N, 3)
        safety_distances = LSE_max(safety_distances, dim=2, rho=10)  # (B, N)
        # Make sure all tensors are (B, Nc_i)
        safety_constraints = safety_distances.view(B, -1)
        kappa_constraints = kappa_violations.view(B, -1)
        distance_constraints = distance_violations.view(B, -1)

        constraints = torch.cat(
            [safety_constraints, kappa_constraints, distance_constraints],
            dim=1
        )

        return constraints  # (B, Nc), all should satisfy <= 0

    return constraints_fn

def get_map_distance(distance_map, grid_x, grid_y, config):
    B, W, H = distance_map.shape
    B, N = grid_x.shape
    grid_x_no_grad = grid_x.detach() # (B, N)
    grid_y_no_grad = grid_y.detach() # (B, N)
    
    x_round = torch.round(grid_x_no_grad).long()
    y_round = torch.round(grid_y_no_grad).long()
    # Eight neighboring directions plus the center point, for nine points total
    x_offsets = torch.tensor([-1, -1, -1, 0, 0, 1, 1, 1, 0], device=grid_x.device).view(1, 1, 9)  # (1, 1, 9)
    y_offsets = torch.tensor([-1, 0, 1, -1, 1, -1, 0, 1, 0], device=grid_y.device).view(1, 1, 9)  # (1, 1, 9)
    x_neighbors = x_round.unsqueeze(2) + x_offsets  # (B, N, 9)
    y_neighbors = y_round.unsqueeze(2) + y_offsets  # (B, N, 9)
    x_neighbors = torch.clamp(x_neighbors, 0, W-1)
    y_neighbors = torch.clamp(y_neighbors, 0, H-1)
    neighbor_potential = distance_map[torch.arange(B).unsqueeze(1).unsqueeze(2), x_neighbors, y_neighbors]  # (B, N, 9)
    min_indices = torch.argmin(neighbor_potential, dim=-1)  # (B, N)
    batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)
    point_indices = torch.arange(N).unsqueeze(0).expand(B, N)
    min_dist_x = x_neighbors[batch_indices, point_indices, min_indices]  # (B, N)
    min_dist_y = y_neighbors[batch_indices, point_indices, min_indices]  # (B, N)
    min_potential = neighbor_potential[batch_indices, point_indices, min_indices]  # (B, N)
    distances_with_min_point_8 = torch.sqrt((grid_x - min_dist_x.float())**2 + (grid_y - min_dist_y.float())**2)  # (B, N)
    
    reformulated_potential = neighbor_potential - min_potential.unsqueeze(2)  # (B, N, 9)
    reformulated_potential = reformulated_potential ** 2
    distance_with_neighbors = torch.sqrt((grid_x.unsqueeze(2) - x_neighbors.float())**2 + (grid_y.unsqueeze(2) - y_neighbors.float())**2)  # (B, N, 9)
    # Make repulsive_force decay rapidly with distance
    repulsive_force = reformulated_potential * torch.exp(-10.0 * distance_with_neighbors)  # (B, N, 9)
    repulsive_force = repulsive_force.sum(dim=2)  # (B, N)

    x_down = torch.floor(grid_x_no_grad).long()
    y_down = torch.floor(grid_y_no_grad).long()
    x_up = x_down + 1
    y_up = y_down + 1
    x_up = torch.clamp(x_up, 0, W-1)
    x_down = torch.clamp(x_down, 0, W-1)
    y_up = torch.clamp(y_up, 0, H-1)
    y_down = torch.clamp(y_down, 0, H-1)
    dist_11 = distance_map[torch.arange(B).unsqueeze(1), x_up, y_up]  # (B, N)
    dist_10 = distance_map[torch.arange(B).unsqueeze(1), x_up, y_down]  # (B, N)
    dist_01 = distance_map[torch.arange(B).unsqueeze(1), x_down, y_up]  # (B, N)
    dist_00 = distance_map[torch.arange(B).unsqueeze(1), x_down, y_down]  # (B, N)
    
    # Stack the distances at the four points (B, N, 4)
    all_dists = torch.stack([dist_00, dist_01, dist_10, dist_11], dim=-1)
    
    # Find the index of the minimum distance (B, N)
    min_indices = torch.argmin(all_dists, dim=-1)
    
    # Select the corresponding coordinates using the minimum-distance index
    batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)
    point_indices = torch.arange(N).unsqueeze(0).expand(B, N)
    
    # Obtain the nearest grid-point coordinates
    
    min_dist = all_dists[batch_indices, point_indices, min_indices]  # (B, N)

    dist_11 = dist_11 - min_dist
    dist_10 = dist_10 - min_dist
    dist_01 = dist_01 - min_dist
    dist_00 = dist_00 - min_dist
    
    mu = 2.0
    dist_11 = dist_11 ** mu
    dist_10 = dist_10 ** mu
    dist_01 = dist_01 ** mu
    dist_00 = dist_00 ** mu
    
    max_delta = 100.0
    dist_11 = torch.clamp(dist_11, max=max_delta)
    dist_10 = torch.clamp(dist_10, max=max_delta)
    dist_01 = torch.clamp(dist_01, max=max_delta)
    dist_00 = torch.clamp(dist_00, max=max_delta)

    dist_11 = dist_11 + min_dist
    dist_10 = dist_10 + min_dist
    dist_01 = dist_01 + min_dist
    dist_00 = dist_00 + min_dist
    
    wa = grid_x - x_down.float()  # (B, N)
    wb = grid_y - y_down.float()  # (B, N)
    # Bilinear interpolation
    distances = (1 - wa) * (1 - wb) * dist_00 + wa * (1 - wb) * dist_10 + \
                (1 - wa) * wb * dist_01 + wa * wb * dist_11  # (B, N)
    # Compute the distance from (grid_x, grid_y) to min_point
    return distances + distances_with_min_point_8 * config['guide_weight']

def world_to_grid(x, y, xmin=globalvar.planning_scale_.xmin, ymin=globalvar.planning_scale_.ymin, resolution=globalvar.planning_scale_.resolution):
    """Convert world coordinates (x, y) to grid indices (i, j)."""
    # i = torch.round((x - xmin) / resolution)
    # j = torch.round((y - ymin) / resolution)
    i = (x - xmin) / resolution
    j = (y - ymin) / resolution
    return i, j

def obj_fn(data, y, config):
    distance_map = data['distance_map']  # (batch_size, H, W)
    xy = y.view(y.shape[0], -1, 7)[:, :, :2]  # (batch_size, N, 2)
    world_x = xy[:, :, 0]  # (B, N) x-coordinates of all points
    world_y = xy[:, :, 1]  # (B, N) y-coordinates of all points
    i, j = world_to_grid(world_x, world_y)  # (B, N)
    distances = get_map_distance(distance_map, i, j, config)  # (B, N)
    map_loss = distances.mean()
    return map_loss
