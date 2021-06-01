import numpy as np
import math
import scipy.spatial

CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
CIRCLE_RADII = [3, 3, 3]  # m


def is_obstacle_frontal(ego_pos, obstacle_pos, ego_yaw, heading_cosine_threshold = (1/(math.sqrt(2)))):
    """[summary]
    It takes in input the position, the yaw angle of the ego vehicle and the position of an obstacle.
    It returns a boolean representing if the the obstacle is in front of the ego vehicle.
    Args:
        ego_pos ((float, float)): ego (x,y) coordinates
        obstacle_pos ((float, float)): obstacle (x,y) coordinates
        ego_yaw (float): ego yaw expressed in radians
    Returns:
        is_in_front (boolean): true if the obstacle is in front of the ego vehicle
    """
    
    obstacle_delta = [obstacle_pos[0] - ego_pos[0], obstacle_pos[1] - ego_pos[1]]
    obstacle_distance = np.linalg.norm(obstacle_delta)
    obstacle_delta = np.divide(obstacle_delta, obstacle_distance)
    
    ego_heading_vector = [math.cos(ego_yaw), math.sin(ego_yaw)]
    
    is_in_front = np.dot(obstacle_delta, ego_heading_vector) > heading_cosine_threshold

    return is_in_front


def project_agent_into_future(agent_x0, agent_y0, agent_yaw, agent_speed, dt, time_to_horizon):
    """[summary]
    It takes in input the position, the yaw, the velocity of an agent,
    the sampling time and the time of horizon.
    It returns a list of the positions that the agent will occupy during the next time_to_horizon seconds
    assuming it moves according to the rectilinear uniform motion model.
    Args:
        agent_x0 (float): agent x coordinate
        agent_y0 (float): agent y coordinate
        agent_yaw (float): agent yaw expressed in radians
        agent_speed (float): agent speed expressed in meters/seconds
        dt (float): sampling time expressed in seconds
        time_to_horizon (float): time of horizon expressed in seconds
    Returns:
        colliding_points (list((float, float))): list of the future locations of the agent
    """
    t = 0
    agent_yaw_rad = math.radians(agent_yaw)
    colliding_points = []
    while t <= time_to_horizon:
        agent_p = next_position(agent_x0, agent_y0, t, agent_speed, agent_yaw_rad)
        colliding_points.append([agent_p['x'], agent_p['y']])
        t = t + dt

    return colliding_points


def next_point(x0, t, speed):
    """[summary]
    It takes in input a coordinate, the time, the velocity.
    It returns the next coordinate assuming it moves according to the rectilinear uniform motion model.
    Args:
        x0 (float): a coordinate
        t (float): time for which the next coordinate must be computed
        speed (float): velocity
    Returns:
        xt (float): next coordinate
    """
    xt = x0 + speed * t
    return xt


def next_position(x0, y0, t, speed, yaw):
    """[summary]
    It takes in input a position, a time, a velocity, a yaw angle.
    It returns the next coordinate assuming it moves according to the rectilinear uniform motion model in the tern
    rotated of -yaw.
    Args:
        x0 (float): x coordinate
        y0 (float): x coordinate
        t (float): time for which the next position must be computed
        speed (float): velocity
        yaw (float): yaw angle expressed in radians of the current tern.
    Returns:
        xt (dict({x, y}): a dict containing the next x and y
    """
    speed_x = speed * math.cos(yaw)
    speed_y = speed * math.sin(yaw)
    x = next_point(x0, t, speed_x)
    y = next_point(y0, t, speed_y)
    return {'x': x, 'y': y}


def points_collide(vehicle_p, vehicle_yaw, agent_p):
    """[summary]
        It takes in input the position, the yaw angle of the ego vehicle and the position of an agent.
        It returns the a boolean representing if there is collision.
        Args:
            vehicle_p (float): (x, y) coordinates of the ego vehicle
            vehicle_yaw (float): yaw angle of the ego vehicle expressed in radians.
            agent_p (float): (x, y) coordinates of an agent
        Returns:
        collision_free (boolean): true if there is collision
        """
    circle_locations = np.zeros((len(CIRCLE_OFFSETS), 2))

    circle_offset = np.array(CIRCLE_OFFSETS)
    circle_locations[:, 0] = vehicle_p['x'] + circle_offset * math.cos(vehicle_yaw)
    circle_locations[:, 1] = vehicle_p['y'] + circle_offset * math.sin(vehicle_yaw)

    collision_dists = scipy.spatial.distance.cdist(np.array([[agent_p['x'], agent_p['y']]]), circle_locations)
    collision_dists = np.subtract(collision_dists, CIRCLE_RADII)
    collision_free = not np.any(collision_dists < 0)

    return not collision_free


