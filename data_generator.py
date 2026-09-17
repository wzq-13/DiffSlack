import numpy as np
import math
import random
from random import randint, random as rand
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import globalvar
from collections import deque
from itertools import chain
import multiprocessing
import time
import os
MAX_LEN = 4.0
MIN_LEN = 1.0

DEFAULT_VISUALIZATION_FONT_SIZES = {
    'title': 18,
    'axis_label': 14,
    'tick_label': 12,
    'legend': 12,
    'colorbar_tick': 10,
    'colorbar_label': 12,
}

def triArea(a, b, c):
    return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def grid_to_world(i, j, xmin=globalvar.planning_scale_.xmin, ymin=globalvar.planning_scale_.ymin, resolution=globalvar.planning_scale_.resolution):
    x = xmin + i * resolution
    y = ymin + j * resolution
    return (x, y)

def world_to_grid(x, y, xmin=globalvar.planning_scale_.xmin, ymin=globalvar.planning_scale_.ymin, resolution=globalvar.planning_scale_.resolution):
    i = round((x - xmin) / resolution)
    j = round((y - ymin) / resolution)
    return (i, j)

def obstacle_blowup_quadrilateral(obstacle, blowup_distance):
    poly = np.array(obstacle)
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
    
    new_poly = poly + delta # (4,2)

    return new_poly

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
def check_segment_intersection(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    A, B = p1, p2
    C, D = p3, p4
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
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
    return edges

def check_polygon_intersection(poly1, poly2):
    if poly1.shape != (2,4) or poly2.shape != (2,4):
        raise ValueError("input must be 2x4 numpy arrays")

    # Disjoint axis-aligned bounds prove that neither edge intersection nor
    # containment is possible.  This cheap rejection avoids most of the more
    # expensive segment tests while placing random obstacles.
    if (np.max(poly1[0]) < np.min(poly2[0]) or
        np.max(poly2[0]) < np.min(poly1[0]) or
        np.max(poly1[1]) < np.min(poly2[1]) or
        np.max(poly2[1]) < np.min(poly1[1])):
        return False
    
    if check_edges_intersection(poly1, poly2):
        return True

    # Check containment
    if check_containment(poly1, poly2):
        return True
    
    return False

def h(x, y, polygons_edges, rho=10.0):
    '''
    -x: point's x coordinate
    -y: point's y coordinate
    -polygons_edges: list of edges for obstacle polygons, each edge is a triplet (a, b, c) with shape (m,4,3)
    '''
    all_distances = []
    for edge_set in polygons_edges:
        distances = []
        for a, b, c in edge_set:
            d = (a * x + b * y + c) / np.sqrt(a**2 + b**2)
            distances.append(d)
        all_distances.append(np.min(np.array(distances)))
    h1 = np.max(np.array(all_distances))
    return h1  + 1.25

def _h_batch(x, y, polygons_edges, chunk_size=16384):
    """Vectorized equivalent of ``h`` for broadcastable coordinate arrays."""
    x, y = np.broadcast_arrays(x, y)
    output_shape = x.shape
    x = x.ravel()
    y = y.ravel()
    edges = np.asarray(polygons_edges)
    a = edges[:, :, 0]
    b = edges[:, :, 1]
    c = edges[:, :, 2]
    denominator = np.sqrt(a**2 + b**2)
    result = np.empty(x.size, dtype=np.result_type(x, y, edges, np.float64))

    # Chunking keeps the temporary [points, obstacles, edges] array small when
    # many generator workers run at the same time.
    for start in range(0, x.size, chunk_size):
        stop = min(start + chunk_size, x.size)
        distances = (
            a[None, :, :] * x[start:stop, None, None]
            + b[None, :, :] * y[start:stop, None, None]
            + c[None, :, :]
        ) / denominator[None, :, :]
        result[start:stop] = np.max(np.min(distances, axis=2), axis=1) + 1.25

    return result.reshape(output_shape)

def check_edges_intersection(poly1, poly2):
    """Check if the edges of two polygons intersect."""
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
    """Check if two line segments intersect."""
    # Ensure all points are 2D coordinates
    a1 = np.asarray(a1).flatten()[:2]
    a2 = np.asarray(a2).flatten()[:2]
    b1 = np.asarray(b1).flatten()[:2]
    b2 = np.asarray(b2).flatten()[:2]
    
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    case1 = ccw(a1, b1, b2) != ccw(a2, b1, b2)
    case2 = ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    if case1 and case2:
        return True
    
    if (np.array_equal(a1, b1) or np.array_equal(a1, b2) or 
        np.array_equal(a2, b1) or np.array_equal(a2, b2)):
        return True
    
    if is_point_on_segment(a1, b1, b2) or is_point_on_segment(a2, b1, b2):
        return True
    if is_point_on_segment(b1, a1, a2) or is_point_on_segment(b2, a1, a2):
        return True
    
    return False

def is_point_on_segment(p, a, b):
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
    if all(point_in_polygon(poly1[:,i], poly2) for i in range(4)):
        return True
    
    if all(point_in_polygon(poly2[:,i], poly1) for i in range(4)):
        return True
    
    return False

def point_in_polygon(point, polygon):
    x, y = point
    n = 4
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

class VehiclePolygon:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.polygon = self._create_polygon()
        
    def _create_polygon(self):
        length = globalvar.vehicle_geometrics_.vehicle_length
        width = globalvar.vehicle_geometrics_.vehicle_width
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        corners = np.array([
            [self.x + length/2*cos_theta - width/2*sin_theta, 
             self.y + length/2*sin_theta + width/2*cos_theta],
            [self.x + length/2*cos_theta + width/2*sin_theta,
             self.y + length/2*sin_theta - width/2*cos_theta],
            [self.x - length/2*cos_theta + width/2*sin_theta,
             self.y - length/2*sin_theta - width/2*cos_theta],
            [self.x - length/2*cos_theta - width/2*sin_theta,
             self.y - length/2*sin_theta + width/2*cos_theta]
        ])
        return corners

def CreateVehiclePolygon(x, y, theta):
    length = globalvar.vehicle_geometrics_.vehicle_length
    width = globalvar.vehicle_geometrics_.vehicle_width
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    corners = np.array([
        [x + length/2*cos_theta - width/2*sin_theta, 
         y + length/2*sin_theta + width/2*cos_theta],
        [x + length/2*cos_theta + width/2*sin_theta,
         y + length/2*sin_theta - width/2*cos_theta],
        [x - length/2*cos_theta + width/2*sin_theta,
         y - length/2*sin_theta - width/2*cos_theta],
        [x - length/2*cos_theta - width/2*sin_theta,
         y - length/2*sin_theta + width/2*cos_theta]
    ])
    return VehiclePolygon(corners[:,0], corners[:,1], theta)

def visualize_environment(planning_scale, vehicle_TPBV, obstacles, show=True, save_path=None):
    fig = plt.figure(figsize=(10, 10))
    
    plt.plot([planning_scale['xmin'], planning_scale['xmax']], 
             [planning_scale['ymin'], planning_scale['ymin']], 'k-')
    plt.plot([planning_scale['xmin'], planning_scale['xmax']], 
             [planning_scale['ymax'], planning_scale['ymax']], 'k-')
    plt.plot([planning_scale['xmin'], planning_scale['xmin']], 
             [planning_scale['ymin'], planning_scale['ymax']], 'k-')
    plt.plot([planning_scale['xmax'], planning_scale['xmax']], 
             [planning_scale['ymin'], planning_scale['ymax']], 'k-')
    
    V_initial = VehiclePolygon(vehicle_TPBV['x0'], vehicle_TPBV['y0'], vehicle_TPBV['theta0'])
    V_terminal = VehiclePolygon(vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], vehicle_TPBV['thetatf'])
    from matplotlib.patches import Polygon
    initial_poly = Polygon(V_initial.polygon, closed=True, fill=True, color='green', alpha=0.5, label='Initial Position')
    terminal_poly = Polygon(V_terminal.polygon, closed=True, fill=True, color='blue', alpha=0.5, label='Terminal Position')
    
    ax = plt.gca()
    ax.add_patch(initial_poly)
    ax.add_patch(terminal_poly)
    
    for i, obs in enumerate(obstacles):
        poly = Polygon(np.column_stack((obs['x'], obs['y'])), closed=True, 
                      fill=True, color='red', alpha=0.3)
        ax.add_patch(poly)
        
        centroid_x = sum(obs['x']) / 4
        centroid_y = sum(obs['y']) / 4
        plt.text(centroid_x, centroid_y, str(i+1), ha='center', va='center', color='black')
    
    for obs in obstacles:
        margin = obs.get('margin', 2.5)
        vertices = np.column_stack((obs['x'], obs['y']))
        margin_vertices = obstacle_blowup_quadrilateral(vertices, margin)
        margin_vertices = np.vstack((margin_vertices, margin_vertices[0]))
        plt.plot(
            margin_vertices[:, 0], margin_vertices[:, 1],
            'r--', linewidth=0.5,
        )
    
    plt.title('Generated Obstacles Environment')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.xlim(planning_scale['xmin'] - 5, planning_scale['xmax'] + 5)
    plt.ylim(planning_scale['ymin'] - 5, planning_scale['ymax'] + 5)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    return fig

def GenerateStaticObstacles_unstructured(planning_scale, vehicle_TPBV, vehicle_geometrics, Nobs):
    lx = globalvar.planning_scale_.obs_x_min
    ux = globalvar.planning_scale_.obs_x_max
    ly = globalvar.planning_scale_.obs_y_min
    uy = globalvar.planning_scale_.obs_y_max

    V_initial = CreateVehiclePolygon(vehicle_TPBV['x0'], vehicle_TPBV['y0'], vehicle_TPBV['theta0'])
    V_terminal = CreateVehiclePolygon(vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], vehicle_TPBV['thetatf'])

    s_k = (vehicle_TPBV['ytf'] - vehicle_TPBV['y0']) / (vehicle_TPBV['xtf'] - vehicle_TPBV['x0'] + 1e-6)
    s_b = vehicle_TPBV['y0'] - s_k * vehicle_TPBV['x0']
    
    obstacles = []
    obj = None
    margin = 2.5
    count = 0
    max_attempts = 100 * Nobs
    
    while count < Nobs and max_attempts > 0:
        max_attempts -= 1
        
        # Generate a random quadrilateral
        while True:
            x = (ux - lx) * rand() + lx
            if count == 1:
                y = s_k * x + s_b + 1 + rand() 
            elif count == 2:
                y = -1 * rand() + uy
            elif count == 3:
                y = 1 * rand() + ly
            else:
                y = (uy - ly) * rand() + ly
            theta = 2 * math.pi * rand() - math.pi
            
            xru = x + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.cos(theta)
            yru = y + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.sin(theta)
            xrd = xru + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.sin(theta)
            yrd = yru - (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.cos(theta)
            xld = x + (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.sin(theta)
            yld = y - (rand() * (MAX_LEN - MIN_LEN) + MIN_LEN) * math.cos(theta)

            if (xru < lx or xru > ux or xrd < lx or xrd > ux or 
                xld < lx or xld > ux or yru < ly or yru > uy or 
                yrd < ly or yrd > uy or yld < ly or yld > uy):
                continue
                
            temp_obj = np.array([[x, xru, xrd, xld], [y, yru, yrd, yld]])
            temp_obj_margin  = obstacle_blowup_quadrilateral(temp_obj.T, margin).T
            
            if is_simple_polygon(temp_obj):
                break
        
        xv = temp_obj_margin[0, :].tolist() + [temp_obj_margin[0, 0]]
        yv = temp_obj_margin[1, :].tolist() + [temp_obj_margin[1, 0]]
        
        if (inpolygon(vehicle_TPBV['x0'], vehicle_TPBV['y0'], xv, yv) or 
            inpolygon(vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], xv, yv)):
            continue
        
        vehicle_poly = np.array([V_initial.x, V_initial.y])#.T
        if check_polygon_intersection(temp_obj_margin, vehicle_poly):
            continue
            
        vehicle_poly = np.array([V_terminal.x, V_terminal.y])#.T
        if check_polygon_intersection(temp_obj_margin, vehicle_poly):
            continue
        
        collision = False
        if obj is not None:
            n = obj.shape[1] // 4
            for i in range(n):
                existing_obs = obj[:, 4*i:4*i+4]
                if check_polygon_intersection(temp_obj_margin, existing_obs):
                    collision = True
                    break
        if collision:
            continue
        
        obstacle = {
            'x': temp_obj[0, :].tolist(),
            'y': temp_obj[1, :].tolist(),
        }
        obstacles.append(obstacle)
        
        if obj is None:
            obj = temp_obj.copy()
        else:
            obj = np.hstack((obj, temp_obj))
        
        count += 1
    
    if count < Nobs:
        raise RuntimeError("can not generate enough obstacles")
    
    return obstacles

def generate_navigation_graph(obstacles_vertices, obstacles_numpy, car_width, resolution, target_node, init_node):
    obstacles = [Polygon(vertices) for vertices in obstacles_vertices]
    obstacle_edges = polygon_edges(obstacles_numpy)
    min_x, max_x = globalvar.planning_scale_.xmin, globalvar.planning_scale_.xmax
    min_y, max_y = globalvar.planning_scale_.ymin, globalvar.planning_scale_.ymax
    xs = np.arange(min_x, max_x + resolution, resolution)
    ys = np.arange(min_y, max_y + resolution, resolution)
    W = len(xs)
    H = len(ys)
    valid_points = {}
    invalid_points = {}
    point_to_index = {}
    index_to_point = {}
    point_index = 0

    # These values used to be calculated point-by-point through ``h``.  The
    # same signed-distance expression is evaluated in batches here, avoiding
    # millions of tiny NumPy allocations per generated sample.
    grid_h = _h_batch(xs[:, None], ys[None, :], obstacle_edges)
    
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            is_inside_obstacle = False
            if abs(y) >= 10 or grid_h[i, j] > 0:
                is_inside_obstacle = True
            
            if not is_inside_obstacle:
                valid_points[(i, j)] = (x, y)
                point_to_index[(i, j)] = point_index
                index_to_point[point_index] = (x, y)
                point_index += 1
            else:
                invalid_points[(i, j)] = (x, y)
    
    graph = {}
    graph_invalid = {}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for grid_pos in valid_points.keys():
        graph[grid_pos] = []
    for grid_pos in invalid_points.keys():
        graph_invalid[grid_pos] = []

    # Cache the three collision samples on every horizontal and vertical grid
    # edge.  Both traversal directions share the same samples, so each edge is
    # evaluated only once.
    horizontal_mid = (xs[:-1] + xs[1:]) / 2
    horizontal_samples = np.stack(
        (
            horizontal_mid,
            (xs[:-1] + horizontal_mid) / 2,
            (horizontal_mid + xs[1:]) / 2,
        ),
        axis=1,
    )
    horizontal_blocked = np.any(
        _h_batch(horizontal_samples[:, None, :], ys[None, :, None], obstacle_edges) > 0,
        axis=2,
    )

    vertical_mid = (ys[:-1] + ys[1:]) / 2
    vertical_samples = np.stack(
        (
            vertical_mid,
            (ys[:-1] + vertical_mid) / 2,
            (vertical_mid + ys[1:]) / 2,
        ),
        axis=1,
    )
    vertical_blocked = np.any(
        _h_batch(xs[:, None, None], vertical_samples[None, :, :], obstacle_edges) > 0,
        axis=2,
    )
    
    distance_dict = {node: -1 for node in valid_points.keys()}
    distance_dict[target_node] = 0
    
    for (i, j), world_coord in valid_points.items():
        for di, dj in directions:
            ni, nj = i + di, j + dj
            neighbor_pos = (ni, nj)
            
            if neighbor_pos in valid_points:
                if di == 1:
                    blocked_by_obstacle = horizontal_blocked[i, j]
                elif di == -1:
                    blocked_by_obstacle = horizontal_blocked[i - 1, j]
                elif dj == 1:
                    blocked_by_obstacle = vertical_blocked[i, j]
                else:
                    blocked_by_obstacle = vertical_blocked[i, j - 1]
                
                # Add only when the edge is not blocked
                if  not blocked_by_obstacle:
                    graph[(i, j)].append(neighbor_pos)
    # BFS queue
    queue = deque([target_node])
    while queue:
        current_node = queue.popleft()
        for neighbor in graph[current_node]:
            if distance_dict[neighbor] == -1:  # Not visited yet
                distance_dict[neighbor] = distance_dict[current_node] + 1
                queue.append(neighbor)

    if distance_dict[init_node] == -1:
        raise ValueError("No path found")
    # For valid points that are still at a distance of 1, change them to invalid points.
    keys_to_delete = []
    for node in distance_dict:
        if distance_dict[node] == -1:
            invalid_points[node] = valid_points[node]
            graph_invalid[node] = []
            keys_to_delete.append(node) # Record the valid key points that need to be deleted
    for key in keys_to_delete:
        del valid_points[key]
        del graph[key]
        del distance_dict[key]
    
    shortest_path = [init_node]
    directions_8 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] #, (1, 1), (1, -1), (-1, 1), (-1, -1)
    current_node = init_node
    while current_node != target_node:
        min_distance = float('inf')
        best_node = None
        for di, dj in directions_8:
            ni, nj = current_node[0] + di, current_node[1] + dj
            neighbor_pos = (ni, nj)
            if neighbor_pos in distance_dict and distance_dict[neighbor_pos] != -1:
                if distance_dict[neighbor_pos] < min_distance:
                    min_distance = distance_dict[neighbor_pos]
                    best_node = neighbor_pos
        shortest_path.append(best_node)
        current_node = best_node
    
    distance_from_shortest_path = {node: -1 for node in valid_points.keys()}
    for node in shortest_path:
        distance_from_shortest_path[node] = 0
    q = deque(shortest_path)
    while q:
        u = q.popleft()
        for neighbor_pos in graph[u]:
            if distance_from_shortest_path[neighbor_pos] == -1:
                distance_from_shortest_path[neighbor_pos] = distance_from_shortest_path[u] + 1
                q.append(neighbor_pos)
    
    for node in valid_points.keys():
        distance_dict[node] = distance_from_shortest_path[node]*2 + distance_dict[node]
        
    distance_map_invalid = {}
    q = deque()
    for (i, j), world_coord in invalid_points.items():
        for di, dj in directions:
            ni, nj = i + di, j + dj
            neighbor_pos = (ni, nj)
            distance_map_invalid[(i, j)] = -1 # Unvisited Flag
            
            if neighbor_pos in valid_points:
                graph_invalid[(i, j)].append(neighbor_pos)
                if not neighbor_pos in graph_invalid: # Valid points adjacent to invalid points that have not been visited before
                    graph_invalid[neighbor_pos] = []
                    distance_map_invalid[neighbor_pos] = distance_dict[neighbor_pos]
                    q.append(neighbor_pos)
                    for di, dj in directions:
                        mi, mj = ni + di, nj + dj
                        neighbor_pos_2 = (mi, mj)
                        if neighbor_pos_2 in invalid_points:
                            graph_invalid[neighbor_pos].append(neighbor_pos_2)
            elif neighbor_pos in invalid_points:
                graph_invalid[(i, j)].append(neighbor_pos)
        
    while q:
        u = q.popleft()
        delta = 10 if u in valid_points else 20
        for neighbor_pos in graph_invalid[u]:
            if distance_map_invalid[neighbor_pos] == -1:
                distance_map_invalid[neighbor_pos] = distance_map_invalid[u] + delta
                q.append(neighbor_pos)
        
    # Delete the valid points in distance map invalid
    distance_map_invalid = {k: v for k, v in distance_map_invalid.items() 
                       if k not in valid_points}

    return graph, valid_points, invalid_points, distance_dict, distance_map_invalid, W, H, shortest_path

def visualize_navigation_graph(obstacles_vertices, graph, valid_points, blocking_lines=None,
                             car_width=None, resolution=None, terminal_point=None,
                             initial_point=None, show=True, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    if blocking_lines is None:
        blocking_lines = []
    if car_width is None:
        car_width = globalvar.vehicle_geometrics_.vehicle_width
    if resolution is None:
        resolution = globalvar.planning_scale_.resolution
    if terminal_point is None:
        raise ValueError("terminal_point must be provided")
    
    for i, vertices in enumerate(obstacles_vertices):
        polygon = Polygon(vertices)
        x, y = polygon.exterior.xy
        ax1.fill(x, y, alpha=0.3, color='pink', label='Obstacles' if i == 0 else "")
        ax1.plot(x, y, 'r--', linewidth=1)
        ax1.text(np.mean(x), np.mean(y), str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax2.fill(x, y, alpha=0.3, color='pink', label='Obstacles' if i == 0 else "")
        ax2.plot(x, y, 'r--', linewidth=1)
        ax2.text(np.mean(x), np.mean(y), str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
    
    for i, line in enumerate(blocking_lines):
        x, y = line.xy
        ax1.plot(x, y, 'g-', linewidth=3, label='Blocking Lines' if i == 0 else "")
        ax2.plot(x, y, 'g-', linewidth=3, label='Blocking Lines' if i == 0 else "")
    
    all_points = list(valid_points.values())
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    
    ax1.scatter(xs, ys, color='blue', s=10, alpha=0.6, label='Grid Points')
    ax2.scatter(xs, ys, color='blue', s=10, alpha=0.6, label='Grid Points')
    
    graph_segments = []
    for grid_pos, neighbors in graph.items():
        p1 = valid_points[grid_pos]
        for neighbor_grid in neighbors:
            if grid_pos >= neighbor_grid:
                continue
            p2 = valid_points[neighbor_grid]
            graph_segments.append((p1, p2))
    if graph_segments:
        from matplotlib.collections import LineCollection
        ax2.add_collection(LineCollection(
            graph_segments, colors='gray', alpha=0.5, linewidths=0.5,
        ))
    edge_count = len(graph_segments)

    if initial_point is not None:
        ax1.plot(initial_point[0], initial_point[1], 'gs', markersize=10, label='Initial Position')
        ax2.plot(initial_point[0], initial_point[1], 'gs', markersize=10, label='Initial Position')
    
    ax1.plot(terminal_point[0], terminal_point[1], 'b^', markersize=10, label='Terminal Position')
    ax2.plot(terminal_point[0], terminal_point[1], 'b^', markersize=10, label='Terminal Position')
    
    ax1.set_xlim(globalvar.planning_scale_.xmin, globalvar.planning_scale_.xmax)
    ax1.set_ylim(globalvar.planning_scale_.ymin, globalvar.planning_scale_.ymax)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Grid Points and Obstacles\n(Car Width: {car_width}m, Resolution: {resolution}m)')
    ax1.legend()
    
    ax2.set_xlim(globalvar.planning_scale_.xmin, globalvar.planning_scale_.xmax)
    ax2.set_ylim(globalvar.planning_scale_.ymin, globalvar.planning_scale_.ymax)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Navigation Graph\n({len(valid_points)} nodes, {edge_count} edges)')
    ax2.legend()
    
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    return fig

def _resolve_visualization_font_sizes(font_sizes):
    resolved_font_sizes = DEFAULT_VISUALIZATION_FONT_SIZES.copy()
    if font_sizes is not None:
        unknown_keys = set(font_sizes) - set(resolved_font_sizes)
        if unknown_keys:
            raise ValueError(
                f"Unknown font size settings: {sorted(unknown_keys)}; "
                f"supported settings are {sorted(resolved_font_sizes)}"
            )
        resolved_font_sizes.update(font_sizes)
    return resolved_font_sizes

def _style_scene_axis(ax, font_family, font_sizes):
    ax.set_xlim(globalvar.planning_scale_.xmin, globalvar.planning_scale_.xmax)
    ax.set_ylim(globalvar.planning_scale_.ymin, globalvar.planning_scale_.ymax)
    ax.set_xlabel(
        'X (m)',
        fontfamily=font_family,
        fontsize=font_sizes['axis_label'],
    )
    ax.set_ylabel(
        'Y (m)',
        fontfamily=font_family,
        fontsize=font_sizes['axis_label'],
    )
    ax.tick_params(axis='both', labelsize=font_sizes['tick_label'])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(font_family)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

def visualize_generated_data(
    data,
    initial_point=None,
    show=True,
    save_path=None,
    font_family='Arial',
    font_sizes=None,
    dijkstra_color="#ACA9A9DD",
    dijkstra_linewidth=1.5,
    dijkstra_alpha=0.75,
):
    """Visualize one generated scene and its potential field in separate figures."""
    resolved_font_sizes = _resolve_visualization_font_sizes(font_sizes)

    obstacles_vertices = np.asarray(data['obstacles_vertices'])
    distance_map = np.asarray(data['distance_map'])
    terminal_point = np.asarray(data['target'])
    if initial_point is None:
        initial_point = np.array([
            globalvar.vehicle_TPBV_.x0,
            globalvar.vehicle_TPBV_.y0,
        ])
    else:
        initial_point = np.asarray(initial_point)

    expected_shape = (
        round((globalvar.planning_scale_.xmax - globalvar.planning_scale_.xmin)
              / globalvar.planning_scale_.resolution) + 1,
        round((globalvar.planning_scale_.ymax - globalvar.planning_scale_.ymin)
              / globalvar.planning_scale_.resolution) + 1,
    )
    if distance_map.shape != expected_shape:
        raise ValueError(
            f"distance_map shape must be {expected_shape}, got {distance_map.shape}"
        )

    if 'shortest_path' in data:
        dijkstra_path = np.asarray(data['shortest_path']).copy()
    else:
        dijkstra_path = _extract_shortest_path_from_potential_field(
            distance_map,
            initial_point,
            terminal_point,
        )
    if (dijkstra_path.ndim != 2 or dijkstra_path.shape[1] != 2
            or len(dijkstra_path) < 2):
        raise ValueError(
            "shortest_path must have shape [num_points, 2] and contain at "
            "least two points, "
            f"got {dijkstra_path.shape}"
        )
    if not np.allclose(dijkstra_path[0], initial_point):
        dijkstra_path = np.vstack((initial_point, dijkstra_path))
    else:
        dijkstra_path[0] = initial_point
    if not np.allclose(dijkstra_path[-1], terminal_point):
        dijkstra_path = np.vstack((dijkstra_path, terminal_point))
    else:
        dijkstra_path[-1] = terminal_point

    scene_fig, scene_ax = plt.subplots(figsize=(8, 6))
    field_fig, field_ax = plt.subplots(figsize=(8, 5.2))
    field_fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.90,
        bottom=0.24,
    )
    # [left, bottom, width, height] in normalized figure coordinates.
    field_colorbar_ax = field_fig.add_axes([0.51, 0.10, 0.30, 0.025])

    image = field_ax.imshow(
        distance_map.T,
        origin='lower',
        extent=(
            globalvar.planning_scale_.xmin,
            globalvar.planning_scale_.xmax,
            globalvar.planning_scale_.ymin,
            globalvar.planning_scale_.ymax,
        ),
        aspect='equal',
        cmap='plasma',
        interpolation='nearest',
    )
    colorbar = field_fig.colorbar(
        image,
        cax=field_colorbar_ax,
        orientation='horizontal',
    )
    colorbar.ax.text(
        1.06,
        0.5,
        'Potential cost',
        transform=colorbar.ax.transAxes,
        ha='left',
        va='center',
        fontfamily=font_family,
        fontsize=resolved_font_sizes['colorbar_label'],
    )
    colorbar.ax.tick_params(labelsize=resolved_font_sizes['colorbar_tick'])
    for tick_label in colorbar.ax.get_xticklabels():
        tick_label.set_fontfamily(font_family)

    for obstacle in obstacles_vertices:
        closed = np.vstack((obstacle, obstacle[0]))
        scene_ax.fill(obstacle[:, 0], obstacle[:, 1], color='tab:red', alpha=0.5)
        scene_ax.plot(closed[:, 0], closed[:, 1], color='firebrick', linewidth=1)
        field_ax.fill(obstacle[:, 0], obstacle[:, 1], color='white', alpha=0.7)
        field_ax.plot(closed[:, 0], closed[:, 1], color='firebrick', linewidth=1)

    field_ax.plot(
        dijkstra_path[:, 0],
        dijkstra_path[:, 1],
        color=dijkstra_color,
        linewidth=dijkstra_linewidth,
        linestyle=(0, (4, 4)),
        alpha=dijkstra_alpha,
        label='Dijkstra path',
        zorder=3,
    )

    for ax in (scene_ax, field_ax):
        ax.plot(
            initial_point[0], initial_point[1],
            marker='o', linestyle='None', markersize=10,
            markerfacecolor='green', markeredgecolor='white',
            markeredgewidth=1.5, label='Initial', zorder=5,
        )
        ax.plot(
            terminal_point[0], terminal_point[1],
            marker='*', linestyle='None', markersize=16,
            markerfacecolor='red', markeredgecolor='white',
            markeredgewidth=1.2, label='Target', zorder=5,
        )
        _style_scene_axis(ax, font_family, resolved_font_sizes)

    scene_ax.set_title(
        'Generated obstacle scene',
        fontfamily=font_family,
        fontsize=resolved_font_sizes['title'],
    )
    field_ax.set_title(
        '(a) G-APF supervision',
        fontfamily=font_family,
        fontsize=resolved_font_sizes['title'],
    )
    legend_font = {
        'family': font_family,
        'size': resolved_font_sizes['legend'],
    }
    scene_ax.legend(loc='upper right', prop=legend_font)
    field_handles, field_labels = field_ax.get_legend_handles_labels()
    field_legend_items = dict(zip(field_labels, field_handles))
    field_labels = ['Initial', 'Target', 'Dijkstra path']
    field_handles = [field_legend_items[label] for label in field_labels]
    field_fig.legend(
        field_handles,
        field_labels,
        loc='center',
        ncol=3,
        bbox_to_anchor=(0.28, 0.10),
        prop=legend_font,
        columnspacing=0.8,  # Horizontal spacing between legend entries
        handletextpad=0.4, # Spacing between the icon and text
        handlelength=1.5,  # Length of line-style legend handles
        borderpad=0.3,     # Padding between the content and legend border
    )
    scene_fig.tight_layout()

    if save_path is not None:
        path_root, extension = os.path.splitext(save_path)
        if not extension:
            extension = '.png'
        if extension == '.png':
            scene_fig.savefig(
                f"{path_root}_environment{extension}",
                dpi=160,
                bbox_inches='tight',
            )
            field_fig.savefig(
                f"{path_root}_potential_field{extension}",
                dpi=160,
                bbox_inches='tight',
            )
        elif extension == '.pdf':
            scene_fig.savefig(
                f"{path_root}_environment{extension}",
                bbox_inches='tight',
            )
            field_fig.savefig(
                f"{path_root}_potential_field{extension}",
            )
    if show:
        plt.show()
    return scene_fig, field_fig

def visualize_nmpc_supervision(
    supervision,
    initial_point=None,
    show=True,
    save_path=None,
    font_family='Arial',
    font_sizes=None,
    vehicle_stride=4,
    vehicle_length=None,
    vehicle_width=None,
    path_marker_size=36,
):
    """Plot an NMPC path with vehicle rectangles centered on path points."""
    resolved_font_sizes = _resolve_visualization_font_sizes(font_sizes)
    obstacles_vertices = np.asarray(supervision['obstacles_vertices'])
    path = np.asarray(supervision['path'])
    terminal_point = np.asarray(supervision['target'])

    if obstacles_vertices.ndim != 3 or obstacles_vertices.shape[1:] != (4, 2):
        raise ValueError(
            "obstacles_vertices must have shape [num_obstacles, 4, 2], "
            f"got {obstacles_vertices.shape}"
        )
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path must have shape [num_steps, 2], got {path.shape}")
    if terminal_point.shape != (2,):
        raise ValueError(f"target must have shape [2], got {terminal_point.shape}")

    if initial_point is None:
        initial_point = np.array([
            globalvar.vehicle_TPBV_.x0,
            globalvar.vehicle_TPBV_.y0,
        ])
    else:
        initial_point = np.asarray(initial_point)
    if initial_point.shape != (2,):
        raise ValueError(f"initial_point must have shape [2], got {initial_point.shape}")
    if not isinstance(vehicle_stride, int) or vehicle_stride <= 0:
        raise ValueError("vehicle_stride must be a positive integer")
    if path_marker_size <= 0:
        raise ValueError("path_marker_size must be positive")
    if vehicle_length is None:
        vehicle_length = globalvar.vehicle_geometrics_.vehicle_length
    if vehicle_width is None:
        vehicle_width = globalvar.vehicle_geometrics_.vehicle_width

    figure, ax = plt.subplots(figsize=(8, 5.2))
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.90,
        bottom=0.20,
    )

    for obstacle_number, obstacle in enumerate(obstacles_vertices):
        closed = np.vstack((obstacle, obstacle[0]))
        ax.fill(
            obstacle[:, 0],
            obstacle[:, 1],
            color='#6E6C6C',
            alpha=1.0,
            label='Obstacle' if obstacle_number == 0 else None,
        )
        # ax.plot(closed[:, 0], closed[:, 1], color='#6E6C6C', linewidth=1)

    display_path = np.vstack((initial_point, path))
    if len(display_path) >= 3:
        display_headings = np.empty(len(display_path), dtype=float)
        # Heading at i uses only the points before and after it:
        # p[i + 1] - p[i - 1].
        central_delta = display_path[2:] - display_path[:-2]
        display_headings[1:-1] = np.arctan2(
            central_delta[:, 1],
            central_delta[:, 0],
        )
        # The endpoints have no neighbors on both sides, so reuse the nearest
        # valid central-difference heading rather than a one-sided estimate.
        display_headings[0] = display_headings[1]
        display_headings[-1] = display_headings[-2]
    elif len(display_path) == 2:
        delta = display_path[1] - display_path[0]
        display_headings = np.full(2, math.atan2(delta[1], delta[0]))
    else:
        display_headings = np.zeros(1)

    ax.plot(
        display_path[:, 0],
        display_path[:, 1],
        color='blue',
        linewidth=1.0,
        solid_capstyle='round',
        marker='o',
        markersize=math.sqrt(path_marker_size),
        markerfacecolor='blue',
        markeredgecolor='blue',
        label='NMPC supervision',
        zorder=4,
    )

    local_corners = np.array([
        [vehicle_length / 2, vehicle_width / 2],
        [vehicle_length / 2, -vehicle_width / 2],
        [-vehicle_length / 2, -vehicle_width / 2],
        [-vehicle_length / 2, vehicle_width / 2],
    ])
    vehicle_indices = list(range(0, len(display_path), vehicle_stride))
    if vehicle_indices[-1] != len(display_path) - 1:
        vehicle_indices.append(len(display_path) - 1)
    for vehicle_number, path_index in enumerate(vehicle_indices):
        heading = display_headings[path_index]
        rotation = np.array([
            [math.cos(heading), -math.sin(heading)],
            [math.sin(heading), math.cos(heading)],
        ])
        vehicle_body = local_corners @ rotation.T + display_path[path_index]
        closed_vehicle_body = np.vstack((vehicle_body, vehicle_body[0]))
        # ax.fill(
        #     vehicle_body[:, 0],
        #     vehicle_body[:, 1],
        #     color='#1DB0CA',
        #     alpha=0.08,
        #     zorder=2,
        # )
        ax.plot(
            closed_vehicle_body[:, 0],
            closed_vehicle_body[:, 1],
            color='#1DB0CA',
            linewidth=0.9,
            alpha=0.7,
            label='Vehicle body' if vehicle_number == 0 else None,
            zorder=3,
        )
    ax.plot(
        initial_point[0], initial_point[1],
        marker='o', linestyle='None', markersize=10,
        markerfacecolor='green', markeredgecolor='white',
        markeredgewidth=1.5, label='Initial', zorder=5,
    )
    ax.plot(
        terminal_point[0], terminal_point[1],
        marker='*', linestyle='None', markersize=16,
        markerfacecolor='red', markeredgecolor='white',
        markeredgewidth=1.2, label='Target', zorder=5,
    )

    _style_scene_axis(ax, font_family, resolved_font_sizes)
    ax.set_title(
        '(b) NMPC supervision',
        fontfamily=font_family,
        fontsize=resolved_font_sizes['title'],
    )
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.13),
        ncol=5,
        prop={
            'family': font_family,
            'size': resolved_font_sizes['legend'],
        },
        columnspacing=0.8,
        handletextpad=0.4,
        handlelength=1.5,
        borderpad=0.3,
    )

    if save_path is not None:
        save_extension = os.path.splitext(os.fspath(save_path))[1].lower()
        if save_extension == '.pdf':
            # Keep the nominal 8 x 5.2 inch canvas.  Tight cropping produces
            # different PDF page sizes and causes accidental scaling in AI.
            figure.savefig(save_path)
        else:
            figure.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    return figure
    
def graph_to_adjacency_list(graph, valid_points):
    adjacency_list = {}
    
    for grid_pos, neighbors in graph.items():
        world_coord = valid_points[grid_pos]
        adjacency_list[world_coord] = []
        for neighbor_grid in neighbors:
            neighbor_world = valid_points[neighbor_grid]
            adjacency_list[world_coord].append(neighbor_world)
    
    return adjacency_list

def compute_shortest_distances(graph, valid_points, target_node):
    distance_dict = {node: -1 for node in valid_points.keys()}
    distance_dict[target_node] = 0
    
    # BFS queue
    queue = deque([target_node])
    
    while queue:
        current_node = queue.popleft()
        
        for neighbor in graph[current_node]:
            if distance_dict[neighbor] == -1:  # Not visited
                distance_dict[neighbor] = distance_dict[current_node] + 1
                queue.append(neighbor)
    
    return distance_dict


def dict_map_to_array(grid_dict, W, H, default_value=np.inf):
    """
    Convert a single dictionary mapping into a 2D array of shape [H, W].
    Args:
        grid_dict: Dict[Tuple[int,int], float] mapping (i,j) to distance.
        H, W: Grid height and width.
        default_value: Default fill value for keys absent from the dictionary.
    Returns:
        distance_array: ndarray [H, W], dtype=np.float32
    """
    distance_array = np.full((W, H), default_value, dtype=np.float32)
    for (i, j), dist in grid_dict.items():
        if 0 <= i < W and 0 <= j < H:
            distance_array[i, j] = dist
    # Check whether every position has been filled
    if np.any(distance_array == default_value):
        # for i in range(H):
        #     for j in range(W):
        #         if distance_array[i, j] == default_value:
        #             print(f"Warning: Position ({i}, {j}) was not filled in the distance array.")
        raise ValueError("Some grid positions were not filled in the distance array.")
    return distance_array

def ensure_clockwise_vertices(vertices):
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 2):
        raise ValueError("shape should be [k, 4, 2]")
    
    k = vertices.shape[0]
    
    # Extract four vertices
    p1 = vertices[:, 0]  # [k, 2]
    p2 = vertices[:, 1]  # [k, 2]
    p3 = vertices[:, 2]  # [k, 2]
    p4 = vertices[:, 3]  # [k, 2]

    # Compute the signed area (2 times the area, ignore the 1/2 factor)
    # Formula: sum = (x1y2 - x2y1) + (x2y3 - x3y2) + (x3y4 - x4y3) + (x4y1 - x1y4)
    sum_val = (p1[:, 0] * p2[:, 1] - p2[:, 0] * p1[:, 1] + 
               p2[:, 0] * p3[:, 1] - p3[:, 0] * p2[:, 1] + 
               p3[:, 0] * p4[:, 1] - p4[:, 0] * p3[:, 1] + 
               p4[:, 0] * p1[:, 1] - p1[:, 0] * p4[:, 1])

    # Determine orientation: sum_val < 0 means clockwise, sum_val > 0 means counterclockwise
    # We need clockwise, so if sum_val > 0 (counterclockwise), we need to flip
    need_flip = sum_val > 0  # [k]

    # Create corrected vertices
    corrected_vertices = vertices.copy()
    
    # For quadrilaterals that need to be flipped, reverse the vertex order from [p1, p2, p3, p4] to [p1, p4, p3, p2]
    if np.any(need_flip):
        # Build the reversed order
        flipped_order = np.stack([p1, p4, p3, p2], axis=1)  # [k, 4, 2]
        
        # Use Boolean indexing to select the quadrilaterals to flip
        corrected_vertices[need_flip] = flipped_order[need_flip]
    
    return corrected_vertices

def generate_map_data(return_timings=False):
    theta0 = np.random.uniform(0, math.pi/2.)
    planning_scale = {
        'xmin': globalvar.planning_scale_.xmin, 'xmax': globalvar.planning_scale_.xmax,
        'ymin': globalvar.planning_scale_.ymin, 'ymax': globalvar.planning_scale_.ymax
    }
    
    vehicle_TPBV = {
        'x0': globalvar.vehicle_TPBV_.x0, 'y0': globalvar.vehicle_TPBV_.y0, 'theta0': theta0,
        'xtf': globalvar.vehicle_TPBV_.xtf, 'ytf': globalvar.vehicle_TPBV_.ytf, 'thetatf': globalvar.vehicle_TPBV_.thetatf
    }
    
    vehicle_geometrics = {
        'vehicle_length': globalvar.vehicle_geometrics_.vehicle_length,
        'vehicle_width': globalvar.vehicle_geometrics_.vehicle_width
    }
    
    Nobs = 8  # Number of obstacles to generate

    terminal_point = np.array([vehicle_TPBV['xtf'], vehicle_TPBV['ytf'], vehicle_TPBV['thetatf']])
    initial_point = np.array([vehicle_TPBV['x0'], vehicle_TPBV['y0'], vehicle_TPBV['theta0']])
    obstacle_start = time.perf_counter()
    obstacles = GenerateStaticObstacles_unstructured(
        planning_scale, vehicle_TPBV, vehicle_geometrics, Nobs
    )
    obstacle_time = time.perf_counter() - obstacle_start

    potential_field_start = time.perf_counter()
    obstacles_vertices = [
        list(zip(obs['x'], obs['y'])) for obs in obstacles
    ]
    obstacles_numpy = np.array(obstacles_vertices).reshape(-1, 4, 2)
    car_width = globalvar.vehicle_geometrics_.vehicle_width
    resolution = globalvar.planning_scale_.resolution
    target_node_grid = world_to_grid(terminal_point[0], terminal_point[1])
    initial_node_grid = world_to_grid(initial_point[0], initial_point[1])
    
    graph, valid_points, invalid_points, distance_dict, distance_map_invalid, W, H, shortest_path = generate_navigation_graph(
        obstacles_vertices, obstacles_numpy, car_width*1.1, resolution, target_node_grid, initial_node_grid
    )
    dict1 = distance_dict
    dict2 = distance_map_invalid
    grid_dict = {**dict1, **dict2}
    distance_map = dict_map_to_array(grid_dict, W, H, default_value=np.inf)
    potential_field_time = time.perf_counter() - potential_field_start

    obstacles_pure = chain.from_iterable(obstacles_vertices)
    obstacles_pure = list(chain.from_iterable(obstacles_pure))
    obstacles_pure = np.array(obstacles_pure).reshape(-1, 4, 2)
    obstacles_pure = ensure_clockwise_vertices(obstacles_pure)
    data={
        'obstacles_vertices': obstacles_pure,
        'distance_map': distance_map,
        'target': terminal_point[:2],
        'shortest_path': np.asarray(
            [valid_points[node] for node in shortest_path],
            dtype=float,
        ),
    }
    if return_timings:
        timings = {
            'obstacle_generation': obstacle_time,
            'potential_field_generation': potential_field_time,
        }
        return data, timings
    return data

def _extract_shortest_path_from_potential_field(
    distance_map,
    start_point,
    target_point,
):
    """Recover the grid shortest path retained by the generated potential field."""
    distance_map = np.asarray(distance_map)
    if distance_map.ndim != 2:
        raise ValueError(
            f"distance_map must be a two-dimensional array, got {distance_map.shape}"
        )

    start_node = world_to_grid(start_point[0], start_point[1])
    target_node = world_to_grid(target_point[0], target_point[1])
    width, height = distance_map.shape
    for name, node in (("start", start_node), ("target", target_node)):
        if not (0 <= node[0] < width and 0 <= node[1] < height):
            raise ValueError(f"{name} point maps outside distance_map: {node}")
        if not np.isfinite(distance_map[node]):
            raise ValueError(f"{name} point has no finite potential value")

    # Match the direction priority used when generate_navigation_graph builds
    # its shortest path, so old files without a stored path remain compatible.
    directions = (
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    current_node = start_node
    path_nodes = [current_node]
    visited = {current_node}
    while current_node != target_node:
        current_value = distance_map[current_node]
        best_node = None
        best_value = np.inf
        for di, dj in directions:
            neighbor = (current_node[0] + di, current_node[1] + dj)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            neighbor_value = distance_map[neighbor]
            if np.isfinite(neighbor_value) and neighbor_value < best_value:
                best_value = neighbor_value
                best_node = neighbor

        if best_node is None or best_value >= current_value or best_node in visited:
            raise ValueError(
                "Cannot recover a descending shortest path from distance_map"
            )
        current_node = best_node
        path_nodes.append(current_node)
        visited.add(current_node)

    return np.asarray([grid_to_world(*node) for node in path_nodes], dtype=float)

def resample_path_by_distance(path, spacing=1.0):
    """Resample an xy polyline at a fixed arc-length interval."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError(
            "path must have shape [num_points, 2] and contain at least two points"
        )
    if not np.all(np.isfinite(path)):
        raise ValueError("path must contain only finite values")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a positive finite value")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.r_[True, segment_lengths > 1e-9]
    path = path[keep]
    if len(path) < 2:
        raise ValueError("path has zero total length")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative_distance = np.r_[0.0, np.cumsum(segment_lengths)]
    total_distance = cumulative_distance[-1]
    sample_distances = np.arange(0.0, total_distance, spacing)
    if len(sample_distances) == 0 or total_distance - sample_distances[-1] > 1e-9:
        sample_distances = np.r_[sample_distances, total_distance]
    else:
        sample_distances[-1] = total_distance

    sampled_path = np.column_stack((
        np.interp(sample_distances, cumulative_distance, path[:, 0]),
        np.interp(sample_distances, cumulative_distance, path[:, 1]),
    ))
    sampled_path[0] = path[0]
    sampled_path[-1] = path[-1]
    return sampled_path

def compute_shortest_path_curvature_score(
    shortest_path,
    spacing=1.0,
    return_sampled_path=False,
):
    """Compute the Trainer-compatible score after fixed-distance sampling."""
    sampled_path = resample_path_by_distance(shortest_path, spacing=spacing)
    if len(sampled_path) < 3:
        score = 1.0
        return (score, sampled_path) if return_sampled_path else score

    curvatures = []
    for point_index in range(1, len(sampled_path) - 1):
        point_before = sampled_path[point_index - 1]
        point = sampled_path[point_index]
        point_after = sampled_path[point_index + 1]
        a = np.linalg.norm(point - point_before)
        b = np.linalg.norm(point_after - point)
        c = np.linalg.norm(point_after - point_before)

        if a == 0 or b == 0 or c == 0:
            curvature = 0.0
        elif (abs(a + b - c) < 1e-6
              or abs(b + c - a) < 1e-6
              or abs(c + a - b) < 1e-6):
            curvature = 0.0
        else:
            curvature = np.sqrt(
                (a + b + c)
                * (b + c - a)
                * (c + a - b)
                * (a + b - c)
            ) / (a * b * c)
        curvatures.append(curvature)

    curvatures = np.asarray(curvatures)
    minimum_turning_radius = globalvar.vehicle_kinematics_.min_turning_radius
    radius = 1.0 / (curvatures + 1e-6)
    score_per_point = np.clip(radius / minimum_turning_radius, 0, 1)
    score = float(np.mean(score_per_point))
    return (score, sampled_path) if return_sampled_path else score

def generate_nmpc_supervision(
    data,
    start_pose=None,
    vehicle_config=None,
    nmpc_config=None,
    initial_path=None,
    save_path=None,
):
    """Generate an NMPC label with an optional path-based initialization.

    ``initial_path=None`` preserves NMPC's original straight initialization.
    Pass an xy array for a custom initialization, or ``'potential_field'`` to
    initialize from the shortest path associated with the potential field.
    """
    # Keep CasADi and the training-data dependencies out of the normal map
    # generation path.  They are imported only when NMPC labels are requested.
    from others.test_NMPC import NMPCConfig, NMPCPlanner, VehicleConfig

    if isinstance(data, (str, bytes, os.PathLike)):
        with np.load(data) as loaded_data:
            obstacles_vertices = np.asarray(
                loaded_data['obstacles_vertices']
            ).copy()
            target = np.asarray(loaded_data['target']).copy()
            distance_map = (
                np.asarray(loaded_data['distance_map']).copy()
                if 'distance_map' in loaded_data else None
            )
            stored_shortest_path = (
                np.asarray(loaded_data['shortest_path']).copy()
                if 'shortest_path' in loaded_data else None
            )
    else:
        obstacles_vertices = np.asarray(data['obstacles_vertices']).copy()
        target = np.asarray(data['target']).copy()
        distance_map = (
            np.asarray(data['distance_map']).copy()
            if 'distance_map' in data else None
        )
        stored_shortest_path = (
            np.asarray(data['shortest_path']).copy()
            if 'shortest_path' in data else None
        )

    if obstacles_vertices.ndim != 3 or obstacles_vertices.shape[1:] != (4, 2):
        raise ValueError(
            "obstacles_vertices must have shape [num_obstacles, 4, 2], "
            f"got {obstacles_vertices.shape}"
        )
    if target.shape != (2,):
        raise ValueError(f"target must have shape [2], got {target.shape}")

    if start_pose is None:
        start_pose = np.array([
            globalvar.vehicle_TPBV_.x0,
            globalvar.vehicle_TPBV_.y0,
            0.0,
        ], dtype=float)
    else:
        start_pose = np.asarray(start_pose, dtype=float)
    if start_pose.shape != (3,):
        raise ValueError(f"start_pose must have shape [3], got {start_pose.shape}")

    if vehicle_config is None:
        vehicle_config = VehicleConfig()
    if nmpc_config is None:
        nmpc_config = NMPCConfig(T=40, dt=0.5)

    if isinstance(initial_path, str):
        if initial_path != 'potential_field':
            raise ValueError(
                "string initial_path must be exactly 'potential_field'"
            )
        if stored_shortest_path is not None:
            resolved_initial_path = stored_shortest_path
        elif distance_map is not None:
            resolved_initial_path = _extract_shortest_path_from_potential_field(
                distance_map,
                start_pose[:2],
                target,
            )
        else:
            raise ValueError(
                "potential-field initialization requires shortest_path or "
                "distance_map in data"
            )
    else:
        resolved_initial_path = initial_path

    planner = NMPCPlanner(vehicle_config, nmpc_config)
    success, result = planner.plan(
        start_pose,
        target,
        obstacles_vertices,
        initial_path=resolved_initial_path,
    )
    if not success:
        raise RuntimeError("NMPC failed to generate a supervision path")

    # Match others/test_NMPC.py::generate_labels_for_IL exactly: the initial
    # state is excluded and only the supervised x/y sequence is saved.
    supervision = {
        'path': np.column_stack((result['x'][1:], result['y'][1:])),
        'obstacles_vertices': obstacles_vertices,
        'target': target,
    }

    if save_path is not None:
        save_path = os.fspath(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez(
            save_path,
            path=supervision['path'],
            obstacles_vertices=supervision['obstacles_vertices'],
            target=supervision['target'],
        )

    return supervision

def deal_single_frame(index):
    np.random.seed(index)
    random.seed(index)
    save_path = f'./dataset/{index}.npz'
    globalvar.vehicle_TPBV_.xtf = globalvar.planning_scale_.target_x_min + rand() * (globalvar.planning_scale_.target_x_max - globalvar.planning_scale_.target_x_min)
    globalvar.vehicle_TPBV_.ytf = globalvar.planning_scale_.target_y_min + rand() * (globalvar.planning_scale_.target_y_max - globalvar.planning_scale_.target_y_min)
    target = np.array([globalvar.vehicle_TPBV_.xtf, globalvar.vehicle_TPBV_.ytf])
    
    if os.path.exists(save_path):
        return
    success = False
    while not success:
        try:
            start = time.perf_counter()
            data, timings = generate_map_data(return_timings=True)
            end = time.perf_counter()
            used_time = end - start
            timings['total'] = used_time
            print(
                f"Time taken to generate data for index {index}: "
                f"total={used_time:.6f}s, "
                f"obstacles={timings['obstacle_generation']:.6f}s, "
                f"potential_field={timings['potential_field_generation']:.6f}s"
            )

            # This diagnostic is intentionally evaluated after the generation
            # timer has stopped, so it is excluded from every timing statistic.
            curvature_score, sampled_shortest_path = (
                compute_shortest_path_curvature_score(
                    data['shortest_path'],
                    spacing=1.0,
                    return_sampled_path=True,
                )
            )
            data['shortest_path_1m'] = sampled_shortest_path
            data['shortest_path_curvature_score'] = np.asarray(
                curvature_score,
                dtype=float,
            )
            timings['shortest_path_curvature_score'] = curvature_score
            print(
                f"Shortest-path curvature score for index {index}: "
                f"{curvature_score:.6f}"
            )
            data['target'] = target
            np.savez(save_path, **data)
            success = True
        except Exception as e:
            if e is KeyboardInterrupt:
                raise e
            print(f"Error processing index {index}: {e}")
            # Generate a new random seed
            new_index = index + 10000000 if index < 10000000 else index + randint(1, 10000000)
            index = new_index
            np.random.seed(index)
            random.seed(index)
    return timings

if __name__ == "__main__":
    num = 2000
    begin = 0
    cpu_num = multiprocessing.cpu_count()
    os.makedirs('./dataset', exist_ok=True)
    indexs = list(range(begin, begin + num))
    # indexs = [12367]
    # with multiprocessing.Pool(processes=cpu_num - 1) as pool:
    #     pool.map(deal_single_frame, indexs)

    # with np.load("dataset/12367.npz") as data:
    #     visualize_generated_data(
    #         data,
    #         font_family='Arial',
    #         font_sizes=DEFAULT_VISUALIZATION_FONT_SIZES,
    #         show=False,
    #         save_path='env_visualization.pdf',
    #     )
    # with np.load("dataset/12367.npz") as data:
    #     supervision = generate_nmpc_supervision(
    #         data,
    #         initial_path='potential_field',
    #     )
    #     fig = visualize_nmpc_supervision(
    #         supervision,
    #         font_family='Arial',
    #         font_sizes=DEFAULT_VISUALIZATION_FONT_SIZES,
    #         show=False,
    #         save_path="nmpc_supervision.pdf",
    #         vehicle_stride=1,
    #     )
    #     plt.close(fig)
    
    total_times = []
    obstacle_times = []
    potential_field_times = []
    shortest_path_curvature_scores = []
    for index in indexs:
        timings = deal_single_frame(index)
        if timings is None:
            continue
        total_times.append(timings['total'])
        obstacle_times.append(timings['obstacle_generation'])
        potential_field_times.append(timings['potential_field_generation'])
        shortest_path_curvature_scores.append(
            timings['shortest_path_curvature_score']
        )
    if total_times:
        print(
            "Average time taken to generate data: "
            f"total={np.mean(total_times):.6f}s, "
            f"obstacles={np.mean(obstacle_times):.6f}s, "
            f"potential_field={np.mean(potential_field_times):.6f}s"
        )
        print(
            "Average shortest-path curvature score (1 m spacing): "
            f"{np.mean(shortest_path_curvature_scores):.6f}"
        )
    else:
        print("No new data was generated; all selected files already exist.")
