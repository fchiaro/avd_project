import numpy as np
import math
import scipy.spatial

CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
CIRCLE_RADII = [3, 3, 3]  # m


def is_obstacle_frontal(ego_pos, obstacle_pos, ego_yaw, heading_cosine_threshold = (1/(math.sqrt(2)))):
    """[summary]

    Args:
        ego_pos ([type]): ego (x,y) coordinates
        obstacle_pos ([type]): obstacle (x,y) coordinates
        ego_yaw ([type]): expressed in radians
    """
    
    obstacle_delta = [obstacle_pos[0] - ego_pos[0], obstacle_pos[1] - ego_pos[1]]
    obstacle_distance = np.linalg.norm(obstacle_delta)
    obstacle_delta = np.divide(obstacle_delta, obstacle_distance)
    
    ego_heading_vector = [math.cos(ego_yaw), math.sin(ego_yaw)]
    
    is_in_front = np.dot(obstacle_delta, ego_heading_vector) > heading_cosine_threshold

    return is_in_front


def project_agent_into_future(agent_x0, agent_y0, agent_yaw, agent_speed, dt, time_to_horizon):

    t = 0
    agent_yaw_rad = math.radians(agent_yaw)
    colliding_points = []
    while t <= time_to_horizon:
        agent_p = next_position(agent_x0, agent_y0, t, agent_speed, agent_yaw_rad)
        colliding_points.append([agent_p['x'], agent_p['y']])
        t = t + dt

    return colliding_points

def predict_collision_points(agent_x0, agent_y0, agent_yaw, agent_speed, vehicle_x0, vehicle_y0, vehicle_yaw,
                             vehicle_speed, dt, time_to_horizon,):
    t = 0

    agent_yaw_rad = math.radians(agent_yaw)
    vehicle_yaw_rad = math.radians(vehicle_yaw)

    colliding_points = []

    while t <= time_to_horizon:
        # Stima posizione di agente e veicolo all'istante t

        agent_p = next_position(agent_x0, agent_y0, t, agent_speed, agent_yaw_rad)
        vehicle_p = next_position(vehicle_x0, vehicle_y0, t, vehicle_speed, vehicle_yaw_rad)

        if points_collide(vehicle_p, vehicle_yaw_rad, agent_p):
            colliding_points.append([agent_p['x'], agent_p['y']])

        t = t + dt

    return colliding_points


def next_point(x0, t, speed):
    return x0 + speed * t


def next_position(x0, y0, t, speed, yaw):
    speed_x = speed * math.cos(yaw)
    speed_y = speed * math.sin(yaw)
    x = next_point(x0, t, speed_x)
    y = next_point(y0, t, speed_y)
    return {'x': x, 'y': y}


def points_collide(vehicle_p, vehicle_yaw, agent_p):
    circle_locations = np.zeros((len(CIRCLE_OFFSETS), 2))

    circle_offset = np.array(CIRCLE_OFFSETS)
    circle_locations[:, 0] = vehicle_p['x'] + circle_offset * math.cos(vehicle_yaw)
    circle_locations[:, 1] = vehicle_p['y'] + circle_offset * math.sin(vehicle_yaw)

    collision_dists = scipy.spatial.distance.cdist(np.array([[agent_p['x'], agent_p['y']]]), circle_locations)
    collision_dists = np.subtract(collision_dists, CIRCLE_RADII)
    collision_free = not np.any(collision_dists < 0)

    return not collision_free


# TEST
test_agent_x = 1
test_agent_y = 2
test_agent_yaw = 0
test_agent_speed = 2

test_vehicle_x = 3
test_vehicle_y = 1
test_vehicle_yaw = 90
test_vehicle_speed = 1

print(predict_collision_points(agent_x0=test_agent_x,
                               agent_y0=test_agent_y,
                               agent_yaw=test_agent_yaw,
                               agent_speed=test_agent_speed,
                               vehicle_x0=test_vehicle_x,
                               vehicle_y0=test_vehicle_y,
                               vehicle_yaw=test_vehicle_yaw,
                               vehicle_speed=test_vehicle_speed,
                               dt=1,
                               time_to_horizon=5))
