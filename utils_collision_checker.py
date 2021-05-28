import numpy as np
import math
import scipy.spatial

CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
CIRCLE_RADII = [1.5, 1.5, 1.5]  # m


def predict_collision_points(agent, dt, time_to_horizon, vehicle):
    t = 0

    agent_x0 = agent.transform.location.x
    agent_y0 = agent.transform.location.y
    agent_yaw = agent.transform.rotation.yaw
    agent_speed = agent.forward_speed

    vehicle_x0 = vehicle.transform.location.x
    vehicle_y0 = vehicle.transform.location.y
    vehicle_yaw = vehicle.transform.rotation.yaw
    vehicle_speed = vehicle.forward_speed

    colliding_points = []

    while t < time_to_horizon:
        # Stima posizione di agente e veicolo all'istante t

        agent_p = next_position(agent_x0, agent_y0, t, agent_speed, agent_yaw)
        vehicle_p = next_position(vehicle_x0, vehicle_y0, t, vehicle_speed, vehicle_yaw)

        if points_collide(vehicle_p, vehicle_yaw, agent_p):
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

    return collision_free
