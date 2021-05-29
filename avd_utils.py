import math
from main import obstacle_to_world


def get_lead_vehicle(ego_vehicle, other_vehicles):
    """[summary]

    Args:
        ego_vehicle ([type]): Agent.vehicle object representing ego vehicle (as provided by CARLA).
        other_vehicles ([type]): List of agent.vehicle objects represnting other vehicles (as provided by CARLA).
    """

    # get the coordinates in world frame of ego vehicle's bounding box
    ego_vehicle_box = obstacle_to_world(ego_vehicle.transform.location, ego_vehicle.bounding_box.extent, ego_vehicle.transform.rotation)

    # put previous coordinates into a suitable dictionary
    ego_vehicle_world = _build_world_vehicle_box_dictionary(ego_vehicle_box)

    other_vehicles_world = []

    for vehicle in other_vehicles:
        # get the coordinates in world frame of vehicle's bounding box
        other_vehicle_box = obstacle_to_world(vehicle.transform.location, vehicle.bounding_box.extent, vehicle.transform.rotation)

        # put previous coordinates into a suitable dictionary
        other_vehicle_world = _build_world_vehicle_box_dictionary(other_vehicle_box)

        # choose the boundary point of current vehicle that is closest to ego vehicle's top center point
        min_distance_point = _position_wrt_ego(ego_vehicle_world,other_vehicle_world)

        # if such point is the bottom center point, then the current vehicle leads ego vehicle
        if min_distance_point[0] == 'bc':
            # save current vehicle's transform and distance from ego vehicle's top center
            other_vehicles_world.append(
                {
                    'vehicle': vehicle,
                    'distance': min_distance_point[1]
                }
            )

    # TODO: prima di fare il return, nel for, bisogna assicurarsi di aggiungere alla lista soltanto veicoli che viaggino nella nostra stessa direzione,
    # perché es. un veicolo alla nostra stessa altezza ma che viaggia nella direzione opposta potrebbe avere il bottom center più vicino rispetto ad uno
    # che sta davanti a noi -> facciamo la differenza tra gli angoli di yaw e vediamo se è al di sotto di una certa soglia (es. 45°)

    if len(other_vehicles_world) == 0:
        return None, None

    # return information regarding the vehicle located at minimum distance from ego vehicle
    lead_vehicle = min(other_vehicles_world, key=lambda dictionary:dictionary['distance'])

    return lead_vehicle['vehicle'], lead_vehicle['distance']


def _build_world_vehicle_box_dictionary(box):
    # bl stays for bottom left
    return {
        'bl': box[0],
        'bc': box[1],
        'br': box[2],
        'cr': box[3],
        'tr': box[4],
        'tc': box[5],
        'tl': box[6],
        'cl': box[7]
    }


def _position_wrt_ego(ego, other):
    # distance of ego vehicle's top center w.r.t. other vehicle's top center, center left, bottom center and right center
    distances = {
        'tc': math.sqrt((ego['tc'][0]-other['tc'][0])**2+(ego['tc'][1]-other['tc'][1])**2),
        'cl': math.sqrt((ego['tc'][0]-other['cl'][0])**2+(ego['tc'][1]-other['cl'][1])**2),
        'bc': math.sqrt((ego['tc'][0]-other['bc'][0])**2+(ego['tc'][1]-other['bc'][1])**2),
        'cr': math.sqrt((ego['tc'][0]-other['cr'][0])**2+(ego['tc'][1]-other['cr'][1])**2)
    }

    # return the point, among the ones listed above, which is at minimum distance from ego vehicle's top center, together with
    # its distance
    return min(distances.items(), key=lambda item:item[1])


def closest_traffic_light_distance(measurement_data):
    traffic_ligts = []
    for agent in measurement_data.non_player_agents:
        if agent.HasField('traffic_light'):
            # print(f"Traffic light transform: {agent.traffic_light.transform}") 
            traffic_ligts.append(agent.traffic_light)
    traffic_ligt_distances = []
    for traffic_light in traffic_ligts:
        traffic_ligt_distances.append(_compute_traffic_light_distance(traffic_light, measurement_data.player_measurements))
    
    return min(traffic_ligt_distances)

def _compute_traffic_light_distance(traffic_light_info, ego_info):
    tl_x = traffic_light_info.transform.location.x
    tl_y = traffic_light_info.transform.location.y
    ego_x = ego_info.transform.location.x
    ego_y = ego_info.transform.location.y


    # return math.sqrt((tl_x-ego_x)**2)
    return math.sqrt((tl_x-ego_x)**2+(tl_y-ego_y)**2)