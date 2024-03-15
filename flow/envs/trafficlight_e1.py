"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env
# from flow.core.kernel.vehicle.aimsun import get

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}


class TrafficLightGrid_a_Env(Env):
    """"状态空间有车辆信息"""
    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 4)

        # used during visualization
        self.observed_ids = []
        self.nqi_top=np.zeros((self.rows * self.cols, 1))
        self.nqi_right = np.zeros((self.rows * self.cols, 1))
        self.nqi_bot = np.zeros((self.rows * self.cols, 1))
        self.nqi_left = np.zeros((self.rows * self.cols, 1))


        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GGGrrrrrrrrrGGGrrrrrrrrr")


        self.discrete = env_params.additional_params.get("discrete", True)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(4 ** self.num_traffic_lights)
        else:
            return Box(
                low=0,
                high=3.999,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)


    @property
    def observation_space(self):
        """See class definition."""
        observation_box = Box(
            low=0.,
            high=1,
            shape=(4 * self.num_traffic_lights+2 * 3 * 4 * self.num_observed * self.num_traffic_lights,),
            dtype=np.float32)
        return  observation_box

    def get_state(self):
        """See parent class.

        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        acc and  traffic light information. This is partially observed
        """

        speeds = []
        dist_to_intersec = []
        acc = []
        NQI_mean = []
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        max_accel= max(2.6,7.5)
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        all_observed_ids = []

        for nodes, edges in self.network.node_mapping:           #争取弄出来每个车道上距离路口最近的，self.get_closest_to_intersection_lane(edge, self.num_observed)
            for edge in edges:
                observed_ids = self.get_closest_to_intersection_lane(edge, self.num_observed)
                all_observed_ids += observed_ids
                # check which edges we have so we can always pad in the right
                # positions
                for observed_id in observed_ids:
                    if observed_id != "":
                        speeds += [self.k.vehicle.get_speed(observed_id) / max_speed ]
                        dist_to_intersec += [(self.k.network.edge_length(
                                self.k.vehicle.get_edge(observed_id)) -
                                self.k.vehicle.get_position(observed_id)) / max_dist]
                        acc += [self.k.vehicle.get_accel(observed_id)]
                    elif observed_id=="":
                        speeds += [0]
                        dist_to_intersec += [0]
                        acc += [0]


            NQI=[]
            for edge in edges:
                straight_right=0
                left=0
                vehs=self.k.vehicle.get_ids_by_edge(edge)
                for veh in vehs:
                    if self.k.vehicle.get_speed(veh) == 0.0:
                        if self.k.vehicle.get_lane(veh) == 0 or self.k.vehicle.get_lane(veh) == 1:
                            straight_right += 1
                        elif self.k.vehicle.get_lane(veh) == 2:
                            left += 1
                NQI.append(straight_right/0.19/self.k.network.edge_length(edge)/2)
                NQI.append(left / 0.19 / self.k.network.edge_length(edge))
            NQI_mean += [((NQI[0] + NQI[4]) / 2)]
            NQI_mean += [((NQI[1] + NQI[5]) / 2)]
            NQI_mean += [((NQI[2] + NQI[6]) / 2)]
            NQI_mean += [((NQI[3] + NQI[7]) / 2)]

        for i in range(len(NQI_mean)//4):
            self.nqi_top[i]=NQI_mean[i*4+0]
            self.nqi_right[i] =NQI_mean[i*4+1]
            self.nqi_bot[i] = NQI_mean[i*4+2]
            self.nqi_left[i]= NQI_mean[i*4+3]

        self.observed_ids = all_observed_ids
        return np.array(
            np.concatenate([
                NQI_mean,speeds, dist_to_intersec,
            ]))

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:       #################################################还有问题
            # convert single value to list of 0's ,1',2's and 3's
            rl_mask = [int(x) for x in list('{}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0 to zero and above 0 to 1. 0 indicates
            # that should not switch the direction, and 1 indicates that switch
            # should happen
            # convert single value to list of 0's ,1',2's and 3's
            rl_mask = rl_actions > 0.0

        for i, action in enumerate(rl_mask):
            if action ==0:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="GGGrrrrrrrrrGGGrrrrrrrrr")
            elif action ==1:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrGGGrrrrrrrrrGGGrrrrrr")
            elif action ==2:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrGGGrrrrrrrrrGGGrrr")
            elif action ==3:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrrrrGGGrrrrrrrrrGGG")

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))


    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.

        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)

        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_closest_to_intersection_lane(self, edges, num_closest, padding=True):

        if num_closest <= 0:
            raise ValueError("Function get_closest_to_intersection_lane called with"
                             "parameter num_closest={}, but num_closest should"
                             "be positive".format(num_closest))

        if isinstance(edges, list):
            ids = [self.get_closest_to_intersection_lane(edge, num_closest)
                   for edge in edges]
            # flatten the list and return it
            return [veh_id for sublist in ids for veh_id in sublist]

        result = []
        lane_0 = []
        lane_1 = []
        lane_2 = []
        all_veh=self.k.vehicle.get_ids_by_edge(edges)
        # print(all_veh)
        # print(self.k.vehicle.get_lane(all_veh))
        for veh in all_veh:
            if self.k.vehicle.get_lane(veh)==0:
                lane_0.append(veh)
            elif self.k.vehicle.get_lane(veh)==1:
                lane_1.append(veh)
            elif self.k.vehicle.get_lane(veh)==2:
                lane_2.append(veh)
        veh_ids_ordered_lane_0=sorted(lane_0, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_0))
        result += (veh_ids_ordered_lane_0[:num_closest] + (pad_lst if padding else []))

        veh_ids_ordered_lane_1 = sorted(lane_1, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_1))
        result += (veh_ids_ordered_lane_1[:num_closest] + (pad_lst if padding else []))

        veh_ids_ordered_lane_2 = sorted(lane_2, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_2))
        result += (veh_ids_ordered_lane_2[:num_closest] + (pad_lst if padding else []))
        # # get the ids of all the vehicles on the edge 'edges' ordered by
        # # increasing distance to end of edge (intersection)
        # lanes=[edges+"_"+str(i) for i in range(3)]                                   #车道数
        # result=[]
        # for lane in lanes:
        #     veh_ids_ordered = sorted(self.k.vehicle.get_ids_by_lane(lane),
        #                              key=self.get_distance_to_intersection)
        #     # return the ids of the num_closest vehicles closest to the
        #     # intersection, potentially with ""-padding.
        #     pad_lst = [""] * (num_closest - len(veh_ids_ordered))
        #     result += (veh_ids_ordered[:num_closest] + (pad_lst if padding else []))
        return result


class TrafficLightGrid_b_Env(Env):
    """"状态空间无车辆信息"""
    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 4)

        # used during visualization
        self.observed_ids = []
        self.nqi_top=np.zeros((self.rows * self.cols, 1))
        self.nqi_right = np.zeros((self.rows * self.cols, 1))
        self.nqi_bot = np.zeros((self.rows * self.cols, 1))
        self.nqi_left = np.zeros((self.rows * self.cols, 1))


        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GGGrrrrrrrrrGGGrrrrrrrrr")


        self.discrete = env_params.additional_params.get("discrete", True)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(4 ** self.num_traffic_lights)
        else:
            return Box(
                low=0,
                high=3.999,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)


    @property
    def observation_space(self):
        """See class definition."""
        observation_box = Box(
            low=0.,
            high=1,
            shape=(4 * self.num_traffic_lights,),
            dtype=np.float32)
        return  observation_box

    def get_state(self):
        """See parent class.

        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        acc and  traffic light information. This is partially observed
        """

        speeds = []
        dist_to_intersec = []
        acc = []
        NQI_mean = []
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        max_accel= max(2.6,7.5)
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        all_observed_ids = []

        for nodes, edges in self.network.node_mapping:           #争取弄出来每个车道上距离路口最近的，self.get_closest_to_intersection_lane(edge, self.num_observed)
            for edge in edges:
                observed_ids = self.get_closest_to_intersection_lane(edge, self.num_observed)
                all_observed_ids += observed_ids
                # check which edges we have so we can always pad in the right
                # positions
                speeds += [
                    self.k.vehicle.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.k.network.edge_length(
                        self.k.vehicle.get_edge(veh_id)) -
                        self.k.vehicle.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]

                acc += [
                    self.k.vehicle.get_accel(veh_id) for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed*3:            #与车道数相对应
                    diff = self.num_observed*3 - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    acc += [0] * diff

            NQI=[]
            for edge in edges:
                straight_right=0
                left=0
                vehs=self.k.vehicle.get_ids_by_edge(edge)
                for veh in vehs:
                    if self.k.vehicle.get_speed(veh) == 0.0:
                        if self.k.vehicle.get_lane(veh) == 0 or self.k.vehicle.get_lane(veh) == 1:
                            straight_right += 1
                        elif self.k.vehicle.get_lane(veh) == 2:
                            left += 1
                NQI.append(straight_right/0.19/self.k.network.edge_length(edge)/2)
                NQI.append(left / 0.19 / self.k.network.edge_length(edge))
            NQI_mean += [((NQI[0] + NQI[4]) / 2)]
            NQI_mean += [((NQI[1] + NQI[5]) / 2)]
            NQI_mean += [((NQI[2] + NQI[6]) / 2)]
            NQI_mean += [((NQI[3] + NQI[7]) / 2)]

        for i in range(len(NQI_mean)//4):
            self.nqi_top[i]=NQI_mean[i*4+0]
            self.nqi_right[i] =NQI_mean[i*4+1]
            self.nqi_bot[i] = NQI_mean[i*4+2]
            self.nqi_left[i]= NQI_mean[i*4+3]

        return np.array(
            NQI_mean
            )

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:       #################################################还有问题
            # convert single value to list of 0's ,1',2's and 3's
            rl_mask = [int(x) for x in list('{}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0 to zero and above 0 to 1. 0 indicates
            # that should not switch the direction, and 1 indicates that switch
            # should happen
            # convert single value to list of 0's ,1',2's and 3's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        for i, action in enumerate(rl_mask):
            if action ==0:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="GGGrrrrrrrrrGGGrrrrrrrrr")
            elif action ==1:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrGGGrrrrrrrrrGGGrrrrrr")
            elif action ==2:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrGGGrrrrrrrrrGGGrrr")
            elif action ==3:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrrrrGGGrrrrrrrrrGGG")

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))

    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.

        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)

        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_closest_to_intersection_lane(self, edges, num_closest, padding=False):

        if num_closest <= 0:
            raise ValueError("Function get_closest_to_intersection_lane called with"
                             "parameter num_closest={}, but num_closest should"
                             "be positive".format(num_closest))

        if isinstance(edges, list):
            ids = [self.get_closest_to_intersection_lane(edge, num_closest)
                   for edge in edges]
            # flatten the list and return it
            return [veh_id for sublist in ids for veh_id in sublist]

        result = []
        lane_0 = []
        lane_1 = []
        lane_2 = []
        all_veh=self.k.vehicle.get_ids_by_edge(edges)
        # print(all_veh)
        # print(self.k.vehicle.get_lane(all_veh))
        for veh in all_veh:
            if self.k.vehicle.get_lane(veh)==0:
                lane_0.append(veh)
            elif self.k.vehicle.get_lane(veh)==1:
                lane_1.append(veh)
            elif self.k.vehicle.get_lane(veh)==2:
                lane_2.append(veh)
        veh_ids_ordered_lane_0=sorted(lane_0, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_0))
        result += (veh_ids_ordered_lane_0[:num_closest] + (pad_lst if padding else []))

        veh_ids_ordered_lane_1 = sorted(lane_1, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_1))
        result += (veh_ids_ordered_lane_1[:num_closest] + (pad_lst if padding else []))

        veh_ids_ordered_lane_2 = sorted(lane_2, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_2))
        result += (veh_ids_ordered_lane_2[:num_closest] + (pad_lst if padding else []))

        return result

class TrafficLightGrid_c_Env(Env):
    """"状态空间无道路信息"""
    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 4)

        # used during visualization
        self.observed_ids = []
        self.nqi_top=np.zeros((self.rows * self.cols, 1))
        self.nqi_right = np.zeros((self.rows * self.cols, 1))
        self.nqi_bot = np.zeros((self.rows * self.cols, 1))
        self.nqi_left = np.zeros((self.rows * self.cols, 1))


        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GGGrrrrrrrrrGGGrrrrrrrrr")


        self.discrete = env_params.additional_params.get("discrete", True)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(4 ** self.num_traffic_lights)
        else:
            return Box(
                low=0,
                high=3.999,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)


    @property
    def observation_space(self):
        """See class definition."""
        observation_box = Box(
            low=0.,
            high=1,
            shape=(2 * 3 * 4 * self.num_observed * self.num_traffic_lights,),
            dtype=np.float32)
        return  observation_box

    def get_state(self):
        """See parent class.

        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        acc and  traffic light information. This is partially observed
        """

        speeds = []
        dist_to_intersec = []
        acc = []
        NQI_mean = []
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        max_accel= max(2.6,7.5)
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        all_observed_ids = []

        for nodes, edges in self.network.node_mapping:           #争取弄出来每个车道上距离路口最近的，self.get_closest_to_intersection_lane(edge, self.num_observed)
            for edge in edges:
                observed_ids = self.get_closest_to_intersection_lane(edge, self.num_observed)
                all_observed_ids += observed_ids
                # check which edges we have so we can always pad in the right
                # positions
                speeds += [
                    self.k.vehicle.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.k.network.edge_length(
                        self.k.vehicle.get_edge(veh_id)) -
                        self.k.vehicle.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]

                acc += [
                    self.k.vehicle.get_accel(veh_id) for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed*3:            #与车道数相对应
                    diff = self.num_observed*3 - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    acc += [0] * diff

            NQI=[]
            for edge in edges:
                straight_right=0
                left=0
                vehs=self.k.vehicle.get_ids_by_edge(edge)
                for veh in vehs:
                    if self.k.vehicle.get_speed(veh) == 0.0:
                        if self.k.vehicle.get_lane(veh) == 0 or self.k.vehicle.get_lane(veh) == 1:
                            straight_right += 1
                        elif self.k.vehicle.get_lane(veh) == 2:
                            left += 1
                NQI.append(straight_right/0.19/self.k.network.edge_length(edge)/2)
                NQI.append(left / 0.19 / self.k.network.edge_length(edge))
            NQI_mean += [((NQI[0] + NQI[4]) / 2)]
            NQI_mean += [((NQI[1] + NQI[5]) / 2)]
            NQI_mean += [((NQI[2] + NQI[6]) / 2)]
            NQI_mean += [((NQI[3] + NQI[7]) / 2)]

        for i in range(len(NQI_mean)//4):
            self.nqi_top[i]=NQI_mean[i*4+0]
            self.nqi_right[i] =NQI_mean[i*4+1]
            self.nqi_bot[i] = NQI_mean[i*4+2]
            self.nqi_left[i]= NQI_mean[i*4+3]

        return np.array(
            np.concatenate([
                speeds, dist_to_intersec,
            ]))

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:       #################################################还有问题
            # convert single value to list of 0's ,1',2's and 3's
            rl_mask = [int(x) for x in list('{}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0 to zero and above 0 to 1. 0 indicates
            # that should not switch the direction, and 1 indicates that switch
            # should happen
            # convert single value to list of 0's ,1',2's and 3's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        for i, action in enumerate(rl_mask):
            if action ==0:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="GGGrrrrrrrrrGGGrrrrrrrrr")
            elif action ==1:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrGGGrrrrrrrrrGGGrrrrrr")
            elif action ==2:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrGGGrrrrrrrrrGGGrrr")
            elif action ==3:
                self.k.traffic_light.set_state(
                    node_id='center{}'.format(i),
                    state="rrrrrrrrrGGGrrrrrrrrrGGG")

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))

    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.

        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)

        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_closest_to_intersection_lane(self, edges, num_closest, padding=False):

        if num_closest <= 0:
            raise ValueError("Function get_closest_to_intersection_lane called with"
                             "parameter num_closest={}, but num_closest should"
                             "be positive".format(num_closest))

        if isinstance(edges, list):
            ids = [self.get_closest_to_intersection_lane(edge, num_closest)
                   for edge in edges]
            # flatten the list and return it
            return [veh_id for sublist in ids for veh_id in sublist]

        result = []
        lane_0 = []
        lane_1 = []
        lane_2 = []
        all_veh=self.k.vehicle.get_ids_by_edge(edges)
        # print(all_veh)
        # print(self.k.vehicle.get_lane(all_veh))
        for veh in all_veh:
            if self.k.vehicle.get_lane(veh)==0:
                lane_0.append(veh)
            elif self.k.vehicle.get_lane(veh)==1:
                lane_1.append(veh)
            elif self.k.vehicle.get_lane(veh)==2:
                lane_2.append(veh)
        veh_ids_ordered_lane_0=sorted(lane_0, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_0))
        result += (veh_ids_ordered_lane_0[:num_closest] + (pad_lst if padding else []))

        veh_ids_ordered_lane_1 = sorted(lane_1, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_1))
        result += (veh_ids_ordered_lane_1[:num_closest] + (pad_lst if padding else []))

        veh_ids_ordered_lane_2 = sorted(lane_2, key=self.get_distance_to_intersection)
        pad_lst = [""] * (num_closest - len(veh_ids_ordered_lane_2))
        result += (veh_ids_ordered_lane_2[:num_closest] + (pad_lst if padding else []))

        return result