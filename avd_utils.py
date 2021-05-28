import math

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