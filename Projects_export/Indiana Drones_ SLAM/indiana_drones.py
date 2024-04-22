######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

"""
 === Introduction ===

   The assignment is broken up into two parts.

   Part A:
        Create a SLAM implementation to process a series of landmark measurements (location of tree centers) and movement updates.
        The movements are defined for you so there are no decisions for you to make, you simply process the movements
        given to you.
        Hint: A planner with an unknown number of motions works well with an online version of SLAM.

    Part B:
        Here you will create the action planner for the drone.  The returned actions will be executed with the goal being to navigate to 
        and extract the treasure from the environment marked by * while avoiding obstacles (trees). 
        Actions:
            'move distance steering'
            'extract treasure_type x_coordinate y_coordinate' 
        Example Actions:
            'move 1 1.570963'
            'extract * 1.5 -0.2'

    Note: All of your estimates should be given relative to your drone's starting location.
    
    Details:
    - Start position
      - The drone will land at an unknown location on the map, however, you can represent this starting location
        as (0,0), so all future drone location estimates will be relative to this starting location.
    - Measurements
      - Measurements will come from trees located throughout the terrain.
        * The format is {'landmark id':{'distance':0.0, 'bearing':0.0, 'type':'D', 'radius':0.5}, ...}
      - Only trees that are within the horizon distance will return measurements. Therefore new trees may appear as you move through the environment.
    - Movements
      - Action: 'move 1.0 1.570963'
        * The drone will turn counterclockwise 90 degrees [1.57 radians] first and then move 1.0 meter forward.
      - Movements are stochastic due to, well, it being a robot.
      - If max distance or steering is exceeded, the drone will not move.
      - Action: 'extract * 1.5 -0.2'
        * The drone will attempt to extract the specified treasure (*) from the current location of the drone (1.5, -0.2).
      - The drone must be within 0.25 distance to successfully extract a treasure.

    The drone will always execute a measurement first, followed by an action.
    The drone will have a time limit of 10 seconds to find and extract all of the needed treasures.
"""
import math
from typing import Dict, List

from rait import matrix

# If you see different scores locally and on Gradescope this may be an indication
# that you are uploading a different file than the one you are executing locally.
# If this local ID doesn't match the ID on Gradescope then you uploaded a different file.
OUTPUT_UNIQUE_FILE_ID = False
if OUTPUT_UNIQUE_FILE_ID:
    import hashlib, pathlib
    file_hash = hashlib.md5(pathlib.Path(__file__).read_bytes()).hexdigest()
    print(f'Unique file ID: {file_hash}')

class SLAM:
    """Create a basic SLAM module.
    """

    def __init__(self):
        """Initialize SLAM components here.
        """
        self.drone_bearing = 0
        self.omega = matrix()
        self.omega.identity(2)
        self.xi = matrix()
        self.xi.zero(2, 1)
        self.mu = matrix()
        self.landmarks = {}

        pass

    # Provided Functions
    def get_coordinates(self):
        """
        Retrieves the estimated (x, y) locations in meters of the drone and all landmarks (trees) when called.

        Args: None

        Returns:
            The (x,y) coordinates in meters of the drone and all landmarks (trees) in the format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        coordinates = {}
        self.mu = self.omega.inverse() * self.xi
        coordinates['self'] = (self.mu[0][0], self.mu[1][0])

        for ID in self.landmarks:
            coordinates[ID] = (self.mu[self.landmarks[ID] * 2][0], self.mu[self.landmarks[ID] * 2 + 1][0])

        return coordinates

    def process_measurements(self, measurements: Dict):
        """
        Process a new series of measurements and update (x,y) location of drone and landmarks

        Args:
            measurements: Collection of measurements of tree positions and radius
                in the format {'landmark id':{'distance': float <meters>, 'bearing':float <radians>, 'type': char, 'radius':float <meters>}, ...}

        """
        for ID in measurements:
            d = measurements[ID]['distance']
            bearing_L = measurements[ID]['bearing']
            theta = bearing_L + self.drone_bearing
            dx = d * math.cos(theta)
            dy = d * math.sin(theta)

            if ID not in self.landmarks:
                l = [i for i in range(2 * (len(self.landmarks) + 1))]
                self.omega = self.omega.expand(self.omega.dimx + 2, self.omega.dimx + 2, l, l)
                self.xi = self.xi.expand(self.xi.dimx + 2, 1, l, [0])
                self.landmarks[ID] = len(self.landmarks) + 1

            # Update X
            self.omega[0][0] += 1 # TODO add uncertainty
            self.omega[0][self.landmarks[ID] * 2] -= 1 # TODO add uncertainty
            self.xi[0][0] -= dx
            self.omega[self.landmarks[ID] * 2][0] -= 1 # TODO add uncertainty
            self.omega[self.landmarks[ID] * 2][self.landmarks[ID] * 2] += 1 # TODO add uncertainty
            self.xi[self.landmarks[ID] * 2][0] += dx

            # Update Y
            self.omega[1][1] += 1 # TODO add uncertainty
            self.omega[1][self.landmarks[ID] * 2 + 1] -= 1 # TODO add uncertainty
            self.xi[1][0] -= dy
            self.omega[self.landmarks[ID] * 2 + 1][1] -= 1 # TODO add uncertainty
            self.omega[self.landmarks[ID] * 2 + 1][self.landmarks[ID] * 2 + 1] += 1 # TODO add uncertainty
            self.xi[self.landmarks[ID] * 2 + 1][0] += dy

        pass


    def process_movement(self, distance: float, steering: float):
        """
        Process a new movement and update (x,y) location of drone

        Args:
            distance: distance to move in meters
            steering: amount to turn in radians
        """
        # insert space for x_t+1 and y_t+1 rows and columns
        l = [0, 1] + [i for i in range(4, self.omega.dimx + 2)]
        self.omega = self.omega.expand(self.omega.dimx + 2, self.omega.dimx + 2, l, l)
        self.xi = self.xi.expand(self.xi.dimx + 2, 1, l, [0])

        # fill those new rows and columns
        dx = distance * math.cos(steering + self.drone_bearing)
        dy = distance * math.sin(steering + self.drone_bearing)

        # Update X
        self.omega[0][0] += 1.0  # TODO add uncertainty
        self.omega[0][2] -= 1.0  # TODO add uncertainty
        self.xi[0][0] -= dx
        self.omega[2][0] -= 1.0  # TODO add uncertainty
        self.omega[2][2] += 1.0  # TODO add uncertainty
        self.xi[2][0] += dx

        # Update Y
        self.omega[1][1] += 1.0  # TODO add uncertainty
        self.omega[1][3] -= 1.0  # TODO add uncertainty
        self.xi[1][0] -= dy
        self.omega[3][1] -= 1.0  # TODO add uncertainty
        self.omega[3][3] += 1.0  # TODO add uncertainty
        self.xi[3][0] += dy

        # # take the subarrays for the next calculation
        # omega_prime = self.omega.take([i for i in range(2, self.omega.dimx)], [i for i in range(2, self.omega.dimx)])
        # xi_prime = self.xi.take([i for i in range(2, self.xi.dimx)], [0])
        # A = self.omega.take([0, 1], [i for i in range(2, self.omega.dimx)])
        # B = self.omega.take([0, 1], [0, 1])
        # C = self.xi.take([0, 1], [0])
        #
        # # calculate new omega and xi
        # self.omega = omega_prime.inverse() - A.transpose() * B.inverse() * A
        # self.xi = xi_prime - A.transpose() * B.inverse() * C
        # self.drone_bearing += steering

        newidxs = list(range(2, len(self.omega.value)))
        a = self.omega.take([0, 1], newidxs)
        b = self.omega.take([0, 1])
        c = self.xi.take([0, 1], [0])
        self.omega = self.omega.take(newidxs) - a.transpose() * b.inverse() * a
        self.xi = self.xi.take(newidxs, [0]) - a.transpose() * b.inverse() * c
        self.drone_bearing += steering
        pass

class IndianaDronesPlanner:
    """
    Create a planner to navigate the drone to reach and extract the treasure marked by * from an unknown start position while avoiding obstacles (trees).
    """

    def __init__(self, max_distance: float, max_steering: float):
        """
        Initialize your planner here.

        Args:
            max_distance: the max distance the drone can travel in a single move in meters.
            max_steering: the max steering angle the drone can turn in a single move in radians.
        """
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.slam = SLAM()

    def next_move(self, measurements: Dict, treasure_location: Dict):
        """Next move based on the current set of measurements.

        Args:
            measurements: Collection of measurements of tree positions and radius in the format 
                          {'landmark id':{'distance': float <meters>, 'bearing':float <radians>, 'type': char, 'radius':float <meters>}, ...}
            treasure_location: Location of Treasure in the format {'x': float <meters>, 'y':float <meters>, 'type': char '*'}
        
        Return: action: str, points_to_plot: dict [optional]
            action (str): next command to execute on the drone.
                allowed:
                    'move distance steering'
                    'move 1.0 1.570963'  - Turn left 90 degrees and move 1.0 distance.
                    
                    'extract treasure_type x_coordinate y_coordinate'
                    'extract * 1.5 -0.2' - Attempt to extract the treasure * from your current location (x = 1.5, y = -0.2).
                                           This will succeed if the specified treasure is within the minimum sample distance.
                   
            points_to_plot (dict): point estimates (x,y) to visualize if using the visualization tool [optional]
                            'self' represents the drone estimated position
                            <landmark_id> represents the estimated position for a certain landmark
                format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        self.slam.process_measurements(measurements)

        coordinates = self.slam.get_coordinates()


        # dist = 1
        # angle = 1

        treasure_x = treasure_location['x']
        treasure_y = treasure_location['y']
        treasure_type = treasure_location['type']

        distance_to_treasure = math.sqrt((coordinates['self'][0] - treasure_x) ** 2 + (coordinates['self'][1] - treasure_y) ** 2)

        # extract if already close enough
        if distance_to_treasure < 0.1: # TODO TUNE
            command = 'extract ' + treasure_type + ' ' + str(coordinates['self'][0]) + ' ' + str(coordinates['self'][1])
            return command, coordinates

        # find all trees which intersect with the path to the treasure
        # find angle

        # danger_trees = []
        for ID in measurements:
            if measurements[ID]['distance'] <= measurements[ID]['radius'] * 1.2:
                pass
                # decide whether to go left or right around it



        dist = min(distance_to_treasure, self.max_distance)

        dx_treasure = treasure_x - coordinates['self'][0]
        dy_treasure = treasure_y - coordinates['self'][1]
        actual_angle = math.atan2(dy_treasure, dx_treasure)
        angle = actual_angle - self.slam.drone_bearing
        # Normalize the angle error to be within -pi to pi radians
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi

        self.slam.process_movement(dist, angle)
        coordinates = self.slam.get_coordinates()
        command = 'move ' + str(dist) + ' ' + str(angle)
        return command, coordinates

def who_am_i():
    # Please specify your GT login ID in the whoami variable (ex: jsmith124).
    whoami = 'apope7'
    return whoami
