#!/usr/bin/env python3
from sys import float_repr_style, stderr
import numpy as np
import math
from avd_utils import get_lead_vehicle
import copy

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
DETECTED_RED_LIGHT = 3
DECELERATE_AND_WAIT = 4
EMERGENCY_STOP = 5
# Stop speed threshold
STOP_THRESHOLD = 0.02
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10


test_counter = 0

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0
        self._TRAFFIC_LIGHT_RED_STATE = 1
        self._TRAFFIC_LIGHT_GREEN_STATE = 0
        self._red_light_count = 0
        self._red_light_count_th = 5 # frames
        self._green_light_count = 0
        self._green_light_count_th = 5 # frames
        self._traffic_light_distance_threshold = 10.0 # meters
        self._stopped = False
        self._n_subsequent_miss = 0
        self._n_subsequent_miss_threshold = 10 # frames
        self._lead_vehicle = None
        self._test_counter = 0
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def _update_goal_state(self, waypoints, ego_state, lookahead=None):
        lookahead_backup = self._lookahead
        if lookahead is not None:
            self._lookahead = lookahead
        # print(f"[Update goal state] lookahead: {self._lookahead}") # DEBUG
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = get_closest_index(waypoints, ego_state)

        # Next, find the goal index that lies within the lookahead distance
        # along the waypoints.
        goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

        # while waypoints[goal_index][2] <= 0.1: goal_index += 1

        self._goal_index = goal_index
        self._goal_state = copy.copy(waypoints[goal_index])

        # print(f"Goal state: {self._goal_state}")

        self._lookahead = lookahead_backup

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed, traffic_light_state, traffic_light_distance,
                         no_path_found=False):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.
        if self._state == FOLLOW_LANE:
            if self._stopped:
                self._stopped = False 
            # print("FOLLOW_LANE")
            self._update_goal_state(waypoints, ego_state)

            if traffic_light_state is not None and traffic_light_distance is not None:
                if traffic_light_state == self._TRAFFIC_LIGHT_RED_STATE:
                    self._state = DETECTED_RED_LIGHT
            if no_path_found:
                self._state = EMERGENCY_STOP

            """
            if 10 <= self._test_counter <= 30:
                print("FRENA")
                self._goal_state[2] = 0
            else:
                print(self._test_counter)
            self._test_counter += 1
            """
        elif self._state == EMERGENCY_STOP:
            self._update_goal_state(waypoints, ego_state)
            self._goal_state[2] = 0
            if not no_path_found:
                self._state = FOLLOW_LANE
        elif self._state == DETECTED_RED_LIGHT:
            # print(f"DETECTED RED LIGHT - {self._red_light_count}") # DEBUG
            self._update_goal_state(waypoints, ego_state)

            if traffic_light_state == self._TRAFFIC_LIGHT_RED_STATE:
                self._red_light_count += 1
            elif traffic_light_state == self._TRAFFIC_LIGHT_GREEN_STATE or traffic_light_state is None:
                self._red_light_count -= 1
            
            if self._red_light_count <= 0:
                self._red_light_count = 0
                self._state = FOLLOW_LANE
            
            if traffic_light_distance is not None and self._red_light_count >= self._red_light_count_th and traffic_light_distance <= self._traffic_light_distance_threshold:
                print("Transitioning to decelerate and wait")
                # impostare come prossimo obiettivo un waypoint vicino al semaforo e mettergli come velocità target zero
                # abbiamo detto che possiamo farlo prendendo un waypoint ad una distanza <= a quella alla quale si trova il semaforo,
                # stesso con la funzione che ci hanno dato loro, e settare la velocità desiderata in quel punto a zero. Per far ciò,
                # abbiamo bisogno di cambiare temporaneamente la distanza di lookahead

                # update goal state by taking the farthest waypoint within the distance between the vehicle and the semaphore
                # self._update_goal_state(waypoints, ego_state, lookahead=traffic_light_distance)
                # closest_len, closest_index = get_closest_index(waypoints, ego_state)
                # self._goal_index = closest_index
                # self._goal_state = waypoints[closest_index]
                # set target speed to zero
                self._goal_state[2] = 0

                self._red_light_count = 0
                self._state = DECELERATE_AND_WAIT
            if no_path_found:
                self._state = EMERGENCY_STOP
                self._red_light_count = 0
        
        elif self._state == DECELERATE_AND_WAIT:
            # print("DECELERATE AND WAIT") # DEBUG
            if traffic_light_state == self._TRAFFIC_LIGHT_GREEN_STATE:
                self._green_light_count += 1

            if traffic_light_state is None or traffic_light_distance is None:
                self._n_subsequent_miss += 1
            else:
                self._n_subsequent_miss = 0
            
            # print(f"Closed loop speed: {abs(closed_loop_speed)}") # DEBUG

            if self._green_light_count >= self._green_light_count_th or self._n_subsequent_miss >= self._n_subsequent_miss_threshold:
                if self._n_subsequent_miss == self._n_subsequent_miss_threshold:
                    print("Transitioning to follow lane due to high number of subsequent miss")
                self._update_goal_state(waypoints, ego_state)
                self._green_light_count = 0
                self._n_subsequent_miss = 0
                self._state = FOLLOW_LANE
            elif abs(closed_loop_speed) <= STOP_THRESHOLD and not self._stopped:
                # reset the count to mitigate the effect of spurious detections
                self._green_light_count = 0
                self._stopped = True

            if no_path_found:
                self._state = EMERGENCY_STOP
                self._green_light_count = 0
                self._n_subsequent_miss = 0
                self._stopped = False

        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            # print("DECELERATE_TO_STOP") # DEBUG
            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                self._stop_count = 0

        # In this state, check to see if we have stayed stopped for at
        # least STOP_COUNTS number of cycles. If so, we can now leave
        # the stop sign and transition to the next state.
        elif self._state == STAY_STOPPED:
            #print("STAY_STOPPED")
            # We have stayed stopped for the required number of cycles.
            # Allow the ego vehicle to leave the stop sign. Once it has
            # passed the stop sign, return to lane following.
            # You should use the get_closest_index(), get_goal_index(), and 
            # check_for_stop_signs() helper functions.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            # We've stopped for the required amount of time, so the new goal 
            # index for the stop line is not relevant. Use the goal index
            # that is the lookahead distance away. 
                            
            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            # If the stop sign is no longer along our path, we can now
            # transition back to our lane following state.
            
            #if not stop_sign_found: self._state = FOLLOW_LANE

            self._state = FOLLOW_LANE
                
        else:
            raise ValueError('Invalid state value.')

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index
        old_arch_length = arc_length

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            # print(f"[Lookahead too small] Distance from the waypoint: {old_arch_length}")
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            # print(f"[End of the path] Distance from the waypoint: {old_arch_length}")
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            old_arch_length = arc_length
            wp_index += 1

        # print(f"[Found waypoint <= lookahead] Distance from the waypoint: {old_arch_length}")

        return wp_index % len(waypoints)
                
    def _filter_leading_vehicles(self, vehicles, ego_state, heading_cosine_threshold):
        leading_vehicles = []
        for vehicle in vehicles:
            vehicle_delta_vector = [vehicle.transform.location.x - ego_state.transform.location.x, 
                                     vehicle.transform.location.y - ego_state.transform.location.y]
            vehicle_distance = np.linalg.norm(vehicle_delta_vector)
            vehicle_delta_vector = np.divide(vehicle_delta_vector, 
                                              vehicle_distance)
            # IL COSENO E' IN RADIANTIIIIIIIIIII
            yaw_radians = ego_state.transform.rotation.yaw/180*math.pi
            ego_heading_vector = [math.cos(yaw_radians), math.sin(yaw_radians)]
            is_in_front = np.dot(vehicle_delta_vector, ego_heading_vector) > heading_cosine_threshold
            is_on_same_direction = abs(vehicle.transform.rotation.yaw-ego_state.transform.rotation.yaw) < 20
            if is_in_front and is_on_same_direction:
                # print(vehicle_delta_vector, '\n', ego_heading_vector)
                # print(f"Ego yaw: {ego_state.transform.rotation.yaw} - vehicle yaw: {vehicle.transform.rotation.yaw}")
                # print(f"internal distance: {vehicle_distance}")
                # print(f"Dot product: {np.dot(vehicle_delta_vector, ego_heading_vector)}")
                leading_vehicles.append(vehicle)
        return leading_vehicles
    

    def _get_closest_vehicle(self, leading_vehicles, ego_state):
        min_distance_vehicle = None
        min_distance = None

        for vehicle in leading_vehicles:
            vehicle_delta_vector = [vehicle.transform.location.x - ego_state.transform.location.x, 
                                     vehicle.transform.location.y - ego_state.transform.location.y]
            vehicle_distance = np.linalg.norm(vehicle_delta_vector)
            if min_distance is None or vehicle_distance < min_distance:
                min_distance = vehicle_distance
                min_distance_vehicle = vehicle
        
        return min_distance_vehicle, min_distance

    
    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, vehicles):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            veicles: A list of agent.vehicle objects of all the vehicles in the world
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).

        returns: 
            agent.vehicle object of vehicle to be followed, None if such vehicle does
            not exist.
        """

        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        
        # if not self._follow_lead_vehicle:
            
        print("Looking for a new leading vehicle...")

        vehicles = self._filter_leading_vehicles(vehicles, ego_state, (1 / math.sqrt(2)))

        # check if there actually are vehicles proceding in our same direction
        if len(vehicles) == 0:
            self._follow_lead_vehicle = False
            return None

        # get closest vehicle that is in front of ego vehicle
        lead_vehicle, lead_vehicle_distance = self._get_closest_vehicle(vehicles, ego_state)

        if lead_vehicle is None or lead_vehicle_distance is None or lead_vehicle_distance > self._follow_lead_vehicle_lookahead:
            self._follow_lead_vehicle = False
            return None

        # self._follow_lead_vehicle = True

        self._lead_vehicle = lead_vehicle

        self._follow_lead_vehicle = True
        
        # print(f"Found new leading vehicle! - {lead_vehicle.id}")
        print(f"Following {lead_vehicle.id}")

        return lead_vehicle

        # else:
        #     print(f"Following the same car - {self._lead_vehicle.id}")

        #     lead_car_delta_vector = [self._lead_vehicle.transform.location.x - ego_state.transform.location.x, 
        #                              self._lead_vehicle.transform.location.y - ego_state.transform.location.y]
        #     lead_car_distance = np.linalg.norm(lead_car_delta_vector)

        #     # Add a 15m buffer to prevent oscillations for the distance check.
        #     if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
        #         return self._lead_vehicle
        #     # Check to see if the lead vehicle is still within the ego vehicle's
        #     # frame of view.
        #     lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
        #     ego_heading_vector = [math.cos(ego_state.transform.rotation.yaw), math.sin(ego_state.transform.rotation.yaw)]
        #     if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
        #         return self._lead_vehicle

        #     self._follow_lead_vehicle = False
            
        #     self._lead_vehicle = None

        #     print("Lost leading vehicle")

        #     return None

# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
