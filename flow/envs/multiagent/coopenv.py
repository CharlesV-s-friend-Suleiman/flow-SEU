"""
Environment for implement of cooperation of trafficlight and autonomous vehicles via reinforcement-learning method
"""
from random import choice
import numpy as np
from gym.spaces import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.multiagent.base import MultiEnv

import math

ADDITIONAL_ENV_PARAMS_CAV = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 15,
}

ADDITIONAL_ENV_PARAMS_TL = {
    # minimum switch time for each traffic light (in seconds), yellow light
    "switch_time": 3.0,
}

ADDITIONAL_ENV_PARAMS_CAVTL = {
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 15,
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 3.0,
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 1,
    # # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
}
# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1

class CustomEnv(MultiEnv):
    """
    A base class for cav-tl

    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        # network.nodemapping is a dict with the following structure:
        #                 mapping[node_id] = [left_edge_id, bot_edge_id,
        #                                     right_edge_id, top_edge_id]
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.get_edge_mappings
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        # after 2 loops, self.lanes_related contains all the lanes around the intersections
        # ['top0_1', 'left1_0', 'right0_0', 'bot0_0', 'bot0_1', 'top0_2',
        # 'left1_1', 'right0_1', 'bot0_2', 'top0_3', 'left1_2', 'right0_2', 'bot0_3',
        # 'left1_3', 'right0_3', 'top0_4', 'top1_1', 'right1_0', 'left2_0', 'bot1_0',
        # 'bot1_1', 'top1_2', 'right1_1', 'left2_1', 'bot1_2', 'top1_3', 'right1_2',
        # 'left2_2', 'bot1_3', 'right1_3', 'left2_3', 'top1_4', 'top2_1', 'right2_0',
        # 'left3_0', 'bot2_0', 'bot2_1', 'top2_2', 'right2_1', 'left3_1', 'bot2_2',
        # 'top2_3', 'right2_2', 'left3_2', 'bot2_3', 'right2_3', 'left3_3', 'top2_4',
        # 'left0_0', 'right3_0', 'left0_1', 'right3_1', 'left0_2', 'right3_2', 'left0_3',
        # 'right3_3', 'top0_0', 'bot0_4', 'top1_0', 'bot1_4', 'top2_0', 'bot2_4', 'bot0_1',
        # 'right1_0', 'left0_0', 'top0_0', 'top0_1', 'bot0_2', 'right1_1', 'left0_1', 'top0_2',
        # 'bot0_3', 'right1_2', 'left0_2', 'top0_3', 'right1_3', 'left0_3', 'bot0_4', 'bot1_1',
        # 'left1_0', 'right2_0', 'top1_0', 'top1_1', 'bot1_2', 'left1_1', 'right2_1', 'top1_2',
        # 'bot1_3', 'left1_2', 'right2_2', 'top1_3', 'left1_3', 'right2_3', 'bot1_4', 'bot2_1',
        # 'left2_0', 'right3_0', 'top2_0', 'top2_1', 'bot2_2', 'left2_1', 'right3_1', 'top2_2',
        # 'bot2_3', 'left2_2', 'right3_2', 'top2_3', 'left2_3', 'right3_3', 'bot2_4', 'right0_0',
        # 'left3_0', 'right0_1', 'left3_1', 'right0_2', 'left3_2', 'right0_3', 'left3_3', 'bot0_0',
        # 'top0_4', 'bot1_0', 'top1_4', 'bot2_0', 'top2_4']
        # traffic light
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights

        # vehicle
        self.observation_info = {}
        self.leader = []
        # vehs list for the edges around intersections in the network
        self.vehs_edges = {i: {} for i in self.lanes_related}
        # used during visualization
        self.observed_ids = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        self.min_switch_time = env_params.additional_params["switch_time"]
        self.target_speed = env_params.additional_params.get("target_velocity", 15)
        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)
        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.tl_type = env_params.additional_params.get('tl_type')
        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GGGrrrrrrrrrGGGrrrrrrrrr")
        self.states_tl = {}
        for node in self.k.traffic_light.get_ids():
            self.states_tl[node] = ["GGGrrrrrrrrrGGGrrrrrrrrr",
                                    "rrrGGGrrrrrrrrrGGGrrrrrr",
                                    "rrrrrrGGGrrrrrrrrrGGGrrr",
                                    "rrrrrrrrrGGGrrrrrrrrrGGG"]

    @property
    def action_space_av(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def action_space_tl(self):
        # the action space for each traffic light is the number of phases
        return Discrete(4)

    @property
    def observation_space_av(self):
        # the observation space for each cav is speed
        # the distance to the intersection, the speed of the leader and follower
        return Box(low=-5, high=5, shape=(9,))

    @property
    def observation_space_tl(self):
        return Box(low=0.,
                   high=1,
                   shape=(4 * self.num_local_edges_max * self.num_observed +
                                          self.num_local_edges_max + self.num_out_edges_max + 3,)
                   )

    @property
    def action_space(self):
        return self.action_space_av, self.action_space_tl

    @property
    def observation_space(self):
        return self.observation_space_av, self.observation_space_tl

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def convert_edge_into_num(self, edge_id):
        return self.lanes_related.index(edge_id) + 1

    def get_observed_info_veh(self, veh_id, max_speed, max_length, accel_norm):
        speed_veh = self.k.vehicle.get_speed(veh_id) / max_speed
        accel_veh = self.k.vehicle.get_realized_accel(veh_id)
        dis_veh = (self.k.network.edge_length(self.k.vehicle.get_edge(veh_id)) -
                   self.k.vehicle.get_position(veh_id)) / max_length
        edge_veh = self.convert_edge_into_num(self.full_name_edge_lane(veh_id)) / len(self.lanes_related)
        # 0: no road to form a 4-leg intersection

        # accel normalization
        if accel_veh < -15:
            accel_veh = -15
        accel_veh = (accel_veh + 15) / accel_norm
        return [speed_veh, accel_veh, dis_veh, edge_veh]

    def get_state(self):
        """See parent class."""
        obs = {}
        self.leader = []

        max_speed = self.k.network.max_speed()
        edge_length = []
        edge_length.extend([self.k.network.edge_length(edge) for edge in self.k.network.get_edge_list()])
        max_length = max(edge_length)

        max_accel, max_deccel = self.env_params.additional_params["max_accel"], 15  # max deccel 15 for emergency stop
        norm_accel = max_accel + max_deccel

        # veh_lane_pair means the vehicles on each lane
        veh_lane_pair = {each: [] for each in self.vehs_edges.keys()}
        for veh_id in self.k.vehicle.get_ids():
            # skip the vehicles not in the lanes_related
            if self.full_name_edge_lane(veh_id) in self.lanes_related:
                veh_lane_pair[self.full_name_edge_lane(veh_id)].append(veh_id)

        # vehicles for incoming and outgoing - info map
        w_max = max_length / 7.5  # normalization for vehicle number, length + min gap
        veh_num_per_edge = {}  # key: name of each edge in the road network
        for each in self.vehs_edges.keys():
            all_vehs = veh_lane_pair[each]
            # remove vehicles already left the edge after one step, not restore at this step
            pre_observed_vehs = list(self.vehs_edges[each].keys())
            for each_veh in pre_observed_vehs:
                if each_veh not in all_vehs:
                    del self.vehs_edges[each][each_veh]
            for veh in all_vehs:
                self.vehs_edges[each].update({veh:self.get_observed_info_veh(veh,max_speed,max_length,norm_accel)})
            if each not in veh_num_per_edge:
                veh_num_per_edge[each] = {}
            veh_num_per_edge[each].update({each: len(self.vehs_edges[each].keys()) / w_max })

        # the veh information observed
        speeds = []
        accels = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []
        # the tl information observed
        for tl_id, edges in self.mapping_inc.items():
            local_speeds = []
            local_accels = []
            local_dist_to_intersec = []
            local_edge_number = []
            for edge in edges:
                # sort the vehicles by distance to the intersection to find the closest ones
                veh_id_sort = {}
                for veh in self.vehs_edges[edge].keys():
                    veh_id_sort.update({self.vehs_edges[edge][veh][2]: veh})  # closer: larger position
                num_observed = min(self.num_observed, len(self.vehs_edges[edge]))
                observed_ids = [veh_id_sort[sorted(veh_id_sort.keys())[i]] for i in range(0, num_observed)]
                all_observed_ids.extend(observed_ids)

                local_speeds.extend([self.vehs_edges[edge][veh_id][0] for veh_id in observed_ids])
                local_accels.extend([self.vehs_edges[edge][veh_id][1] for veh_id in observed_ids])
                local_dist_to_intersec.extend([self.vehs_edges[edge][veh_id][2] for veh_id in observed_ids])
                local_edge_number.extend([self.vehs_edges[edge][veh_id][3] for veh_id in observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_accels.extend([0] * diff)
                    local_dist_to_intersec.extend([1] * diff)
                    local_edge_number.extend([0] * diff)

            speeds.append(local_speeds)
            accels.append(local_accels)
            dist_to_intersec.append(local_dist_to_intersec)
            edge_number.append(local_edge_number)

        self.observed_ids = all_observed_ids  # self.observed_ids contains all the observed vehicles

        # traffic light for each CAV
        incoming_tl = {each:"" for each in self.observed_ids}
        for tl_id in self.k.traffic_light.get_ids():
            tl_id_num = list(self.mapping_inc.keys()).index(tl_id)
            local_edges = self.mapping_inc[tl_id]
            local_edges_out = self.mapping_out[tl_id]
            veh_num_per_in = [veh_num_per_edge[each][each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each][each] for each in local_edges_out]

            for cav_id in incoming_tl.keys():
                if self.full_name_edge_lane(cav_id) in local_edges:
                    incoming_tl[cav_id] = tl_id_num  # get the id of the approaching TL
            states = self.states_tl[tl_id]

            now_state = list(self.k.traffic_light.get_state(tl_id))
            for _ in range(len(now_state)):
                if now_state[_]=='g':
                    now_state[_]='r'
            now_state = ''.join(now_state)
            state_idx = states.index(now_state)

            con = [round(i, 8) for i in np.concatenate(
                [speeds[tl_id_num], accels[tl_id_num], dist_to_intersec[tl_id_num],
                 edge_number[tl_id_num],
                 veh_num_per_in, veh_num_per_out,
                 [self.last_changes[tl_id_num] / 3],
                 [state_idx / len(states)], [self.currently_yellows[tl_id_num]]])]

            observation = np.array(con)
            obs.update({tl_id: observation})
            self.last_changes.append(0)
            self.currently_yellows.append(1)  # if there is no traffic light
        print(incoming_tl )
        # agent_CAV information
        for rl_id in self.observed_ids:
            this_pos = self.k.network.edge_length(self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)
            this_speed = self.k.vehicle.get_speed(rl_id)
            this_accel = self.k.vehicle.get_realized_accel(rl_id)
            this_accel = (this_accel, -15)[abs(this_accel) >= 15]
            lead_id = self.k.vehicle.get_leader(rl_id)

            this_tl_name = ""
            if incoming_tl[rl_id] != "":
                incoming_tl_id = int(incoming_tl[rl_id])
                this_tl_name = list(self.mapping_inc.keys())[incoming_tl_id]
            else:
                incoming_tl_id = -1  # set default value

            if this_tl_name:
                states = self.states_tl[this_tl_name]
                now_state = self.k.traffic_light.get_state(this_tl_name)
                state_index = states.index(now_state)
            else:
                states = []
                state_index = 0

            if states:
                state_norm = state_index / len(states)
            else:
                state_norm = 0

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed + this_speed
                lead_head = max_length
                lead_accel = 15
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_headway(rl_id)
                lead_accel = self.k.vehicle.get_realized_accel(lead_id)
                lead_accel = (lead_accel, -15)[abs(lead_accel) >= 15]

            if lead_head / max_length > 5:
                lead_head = 5 * max_length
            elif lead_head / max_length < -5:
                lead_head = -5 * max_length

            obs.update({rl_id: np.array([
                this_pos / max_length,
                this_speed / max_speed,
                this_accel / max_accel,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                lead_accel / max_accel,
                self.last_changes[incoming_tl_id] / 3,
                state_norm, self.currently_yellows[incoming_tl_id]])})
        # the observation of cav includes the speed, acceleration, distance to the intersection,
        # speed difference between the leader and the follower, headway, leader's acceleration(1-6)
        # the information of the traffic light(7-9)
        self.observation_info = obs
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        reward = {}
        # reward for traffic light
        for rl_id in self.k.traffic_light.get_ids():
            obs = self.observation_info[rl_id]
            # pressure
            traffic_start = 4 * self.num_local_edges_max * self.num_observed
            inc_traffic = np.sum(obs[traffic_start: traffic_start + self.num_local_edges_max])
            out_traffic = np.sum(obs[traffic_start + self.num_local_edges_max:
                                     traffic_start + self.num_local_edges_max + self.num_out_edges_max])
            reward[rl_id] = -(inc_traffic - out_traffic)

        max_speed = self.k.network.max_speed()
        # reward for cav
        for rl_id in self.observed_ids:
            # the reward to forward the vehicles
            rl_speed = sum(self.k.vehicle.get_speed(rl_id)) + .001
            rv = rl_speed / max_speed
            if rl_speed - max_speed >= 5:
                rv -= 10
            reward[rl_id] = rv

        return reward

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            # light
            if rl_id in self.mapping_inc.keys():
                tl_id_num = list(self.mapping_inc.keys()).index(rl_id)
                action = rl_action > 0.0

                states = self.states_tl[rl_id]
                now_state = list(self.k.traffic_light.get_state(rl_id))
                for _ in range(len(now_state)):
                    if now_state[_] == 'g':
                        now_state[_] = 'r'
                now_state = ''.join(now_state)
                state_idx = states.index(now_state)

                if self.currently_yellows[tl_id_num] == 1: # if current yellow then set the last change
                    self.last_changes[tl_id_num] += self.sim_step
                    if round(float(self.last_changes[tl_id_num]), 8) >= self.min_switch_time:
                        if now_state == states[-1]:
                            state_idx = 0
                        else:
                            state_idx += 1
                        self.k.traffic_light.set_state(node_id=rl_id, state=states[state_idx])
                        if 'G' not in states[state_idx]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0
                else:
                    if action: # swich to the next phase
                        if action == 0:
                            self.k.traffic_light.set_state(
                                node_id=rl_id,
                                state="GGGrrrrrrrrrGGGrrrrrrrrr")
                        elif action == 1:
                            self.k.traffic_light.set_state(
                                node_id=rl_id,
                                state="rrrGGGrrrrrrrrrGGGrrrrrr")
                        elif action == 2:
                            self.k.traffic_light.set_state(
                                node_id=rl_id,
                                state="rrrrrrGGGrrrrrrrrrGGGrrr")
                        elif action == 3:
                            self.k.traffic_light.set_state(
                                node_id=rl_id,
                                state="rrrrrrrrrGGGrrrrrrrrrGGG")

                        if 'G' not in states[state_idx]:
                            self.currently_yellows[tl_id_num] = 1
                            self.last_changes[tl_id_num] = 0.0
                        else:
                            self.currently_yellows[tl_id_num] = 0

            else:  # cav
                accel = rl_action[rl_id]
                self.k.vehicle.apply_acceleration(rl_id, accel)

    def reset(self, **kwargs):
        self.leader = []
        self.observation_info = {}
        self.vehs_edges = {i: {} for i in self.lanes_related}
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        return super().reset()

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.k.vehicle.get_ids():
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
        for veh_id in self.observed_ids:
            self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))

class CustomTryEnv(MultiEnv):
    """
    A base class for cav-tl

    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS_CAVTL.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        # network.nodemapping is a dict with the following structure:
        #                 mapping[node_id] = [left_edge_id, bot_edge_id,
        #                                     right_edge_id, top_edge_id]
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.get_edge_mappings
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        self.num_traffic_lights = len(self.mapping_inc.keys())
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights

        # vehicle
        self.observation_info = {}
        self.leader = []
        # vehs list for the edges around intersections in the network
        self.vehs_edges = {i: {} for i in self.lanes_related}
        # used during visualization
        self.observed_ids = []
        # exp setting
        self.num_observed = env_params.additional_params.get("num_observed", 1)
        self.min_switch_time = env_params.additional_params["switch_time"]
        self.target_speed = env_params.additional_params.get("target_velocity", 15)
        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 0)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)
        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.tl_type = env_params.additional_params.get('tl_type')
        self.lastphase = [[0] for i in range(self.num_traffic_lights)]
        self.last_change2green = [[0] for i in range(self.num_traffic_lights)]
        self.last_change2yellow = [[] for i in range(self.num_traffic_lights)]
        self.current_yellow = [[0] for i in range(self.num_traffic_lights)]
        self.nowsecond = [0]

        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GGGrrrrrrrrrGGGrrrrrrrrr")
        self.states_tl = {}
        for node in self.k.traffic_light.get_ids():
            self.states_tl[node] = ["GGGrrrrrrrrrGGGrrrrrrrrr",
                                    "rrrGGGrrrrrrrrrGGGrrrrrr",
                                    "rrrrrrGGGrrrrrrrrrGGGrrr",
                                    "rrrrrrrrrGGGrrrrrrrrrGGG"]

    @property
    def action_space_av(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=self.env_params.additional_params["max_accel"], shape=(1,))

    @property
    def action_space_tl(self):
        # the action space for each traffic light is the number of phases
        return Discrete(4)

    @property
    def observation_space_av(self):
        # the observation space for each cav is speed
        # the distance to the intersection, the speed of the leader and follower
        return Box(low=-100, high=2000, shape=(6,),dtype=np.float32)

    @property
    def observation_space_tl(self):
        return Box(low=0,
                   high=1,
                   shape=(4 * (1 + self.num_local_lights)
                          + 2 * 3 * 4 * self.num_observed,),
                   dtype=np.float32)

    @property
    def action_space(self):
        return self.action_space_av, self.action_space_tl

    @property
    def observation_space(self):
        return self.observation_space_av, self.observation_space_tl

    def full_name_edge_lane(self, veh_id):
        edge_id = self.k.vehicle.get_edge(veh_id)
        lane_id = self.k.vehicle.get_lane(veh_id)
        return edge_id + '_' + str(lane_id)

    def convert_edge_into_num(self, edge_id):
        return self.lanes_related.index(edge_id) + 1

    def get_observed_info_veh(self, veh_id, max_speed, max_length, accel_norm):
        speed_veh = self.k.vehicle.get_speed(veh_id) / max_speed
        accel_veh = self.k.vehicle.get_realized_accel(veh_id)
        dis_veh = (self.k.network.edge_length(self.k.vehicle.get_edge(veh_id)) -
                   self.k.vehicle.get_position(veh_id)) / max_length
        edge_veh = self.convert_edge_into_num(self.full_name_edge_lane(veh_id)) / len(self.lanes_related)
        # 0: no road to form a 4-leg intersection

        # accel normalization
        if accel_veh < -15:
            accel_veh = -15
        accel_veh = (accel_veh + 15) / accel_norm
        return [speed_veh, accel_veh, dis_veh, edge_veh]

    def get_state(self):
        """See parent class."""
        obs = {}

        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()
        max_dist = 3000
        for id in ids:
            veh_pos = []
            veh_v = []
            # get the distance of each rl-veh with the leader and follower
            pos = self.k.vehicle.get_position(id)
            pos /= max_dist
            vel = self.k.vehicle.get_speed(id)
            vel /= max_speed
            veh_pos.append(pos)
            veh_v.append(vel)

            # ids of f and l
            follower = self.k.vehicle.get_follower(id)
            leader = self.k.vehicle.get_leader(id)

            f_veh_pos = self.k.vehicle.get_position(follower)
            if f_veh_pos == -1001:
                f_veh_pos = pos - 50
            f_veh_pos /= max_dist
            l_veh_pos = self.k.vehicle.get_position(leader)
            if l_veh_pos == -1001:
                l_veh_pos = pos + 50
            l_veh_pos /= max_dist
            veh_pos.append(f_veh_pos)
            veh_pos.append(l_veh_pos)

            f_veh_speed = self.k.vehicle.get_speed(follower)
            l_veh_speed = self.k.vehicle.get_speed(leader)
            if f_veh_speed == -1001:
                f_veh_speed = max_speed
            f_veh_speed /= max_speed
            if l_veh_speed == -1001:
                l_veh_speed = max_speed
            l_veh_speed /= max_speed
            veh_v.append(f_veh_speed)
            veh_v.append(l_veh_speed)

            state_cav = np.array(np.concatenate((veh_pos, veh_v)))
            obs.update({id: state_cav})

        speeds = []
        dist_to_intersec = []
        acc = []
        NQI_mean = []
        NQI_mean_0 = []
        all_observed_ids = []
        for nodes, edges in self.network.node_mapping:
            local_speeds = []
            local_dists_to_intersec = []
            local_acc = []
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection_lane(edge, self.num_observed)
                all_observed_ids.append(observed_ids)

                # check which edges we have so we can always pad in the right
                # positions
                for observed_id in observed_ids:
                    if observed_id != "":
                        local_speeds.extend(
                            [self.k.vehicle.get_speed(observed_id) / max_speed])
                        local_dists_to_intersec.extend([abs(self.k.network.edge_length(
                            self.k.vehicle.get_edge(observed_id)) -
                                                         self.k.vehicle.get_position(observed_id)) / max_dist])
                        local_acc.extend([self.k.vehicle.get_accel(observed_id)])
                    elif observed_id == "":
                        local_speeds.extend([0])
                        local_dists_to_intersec.extend([0])
                        local_acc.extend([0])

            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            acc.append(local_acc)

            local_NQI_mean = []
            NQI = []
            for edge in edges:
                straight_right = 0
                left = 0
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                for veh in vehs:
                    if self.k.vehicle.get_speed(veh) == 0.0:
                        if self.k.vehicle.get_lane(veh) == 0 or self.k.vehicle.get_lane(veh) == 1:
                            straight_right += 1
                        elif self.k.vehicle.get_lane(veh) == 2:
                            left += 1
                NQI.append(straight_right / 0.19 / self.k.network.edge_length(edge) / 2)
                NQI.append(left / 0.19 / self.k.network.edge_length(edge))
            local_NQI_mean.extend([((NQI[0] + NQI[4]) / 2)])
            local_NQI_mean.extend([((NQI[1] + NQI[5]) / 2)])
            local_NQI_mean.extend([((NQI[2] + NQI[6]) / 2)])
            local_NQI_mean.extend([((NQI[3] + NQI[7]) / 2)])
            NQI_mean_0.append(local_NQI_mean)
        NQI_mean_0.append([0.0, 0.0, 0.0, 0.0])
        self.observed_ids = all_observed_ids

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_id_nums = [rl_id_num]

            NQI_mean_1 = []
            for local_id_num in local_id_nums:
                NQI_mean_1 += NQI_mean_0[local_id_num]
            NQI_mean.append(NQI_mean_1)

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            state_tl = np.array(np.concatenate(
                [NQI_mean[rl_id_num], speeds[rl_id_num], dist_to_intersec[rl_id_num]]))

            obs.update({rl_id: state_tl})

        return obs

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            # light
            if rl_id in self.mapping_inc.keys():
                i = int(rl_id.split("center")[ID_IDX])
                action = int(rl_action)
                if action == 0:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="GGGrrrrrrrrrGGGrrrrrrrrr")
                elif action == 1:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrGGGrrrrrrrrrGGGrrrrrr")
                elif action == 2:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrrrrGGGrrrrrrrrrGGGrrr")
                elif action == 3:
                    self.k.traffic_light.set_state(
                        node_id='center{}'.format(i),
                        state="rrrrrrrrrGGGrrrrrrrrrGGG")
            else:  # cav
                self.k.vehicle.apply_acceleration(rl_id, rl_action)
        self.nowsecond.append((self.nowsecond[-1] + 1))

    def compute_reward(self, rl_actions, **kwargs):
        if rl_actions is None:
            return {}

        rews = {}
        # reward for traffic light
        NIN = 0
        for nodes, edges in self.network.node_mapping:
            Nin = 0
            for edge in edges:
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                Nin += len(vehs)
            NIN += Nin

        NOUT = 0
        for nodes, edges in self.network.node_mapping_leave:
            Nout = 0
            for edge in edges:
                vehs = self.k.vehicle.get_ids_by_edge(edge)
                Nout += len(vehs)
            NOUT += Nout

        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        veh_ids = self.k.vehicle.get_ids()
        vel = np.array(self.k.vehicle.get_speed(veh_ids))
        penalty = len(vel[vel == 0]) / len(vel)
        NIN /= grid_array["short_length"]
        NOUT /= grid_array["long_length"]
        rew_tl = -(NIN - NOUT) / 3 / 0.15 - penalty
        rew_tl /= self.num_traffic_lights

        for rl_id in rl_actions.keys():
            rews[rl_id] = rew_tl

        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()
        num_rl_veh = self.k.vehicle.num_rl_vehicles
        # the reward to forward the vehicles

        average_rl_speed = (sum(self.k.vehicle.get_speed(ids)) + .001) / (num_rl_veh + .001)

        rew_cav = average_rl_speed / max_speed
        if average_rl_speed - max_speed >= 5:
            rew_cav -= - 10

        # reward for cav
        for rl_id in ids:
            rews[rl_id] = rew_cav

        return rews

    def reset(self, **kwargs):
        self.leader = []
        self.observation_info = {}
        self.vehs_edges = {i: {} for i in self.lanes_related}
        self.last_changes = [0.00] * self.num_traffic_lights
        self.currently_yellows = [0] * self.num_traffic_lights
        return super().reset()

    # def additional_command(self):
    #     # specify observed vehicles
    #     for veh_id in self.k.vehicle.get_ids():
    #         self.k.vehicle.set_color(veh_id=veh_id, color=(255, 255, 255))
    #     for veh_id in self.observed_ids:
    #         self.k.vehicle.set_color(veh_id=veh_id, color=(255, 0, 0))

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

        return result

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

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : dict of array_like
            agent's observation of the current environment
        reward : dict of floats
            amount of reward associated with the previous state/action pair
        done : dict of bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    accel_contr = self.k.vehicle.get_acc_controller(veh_id)
                    action = accel_contr.get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(veh_id)
                    routing_actions.append(route_contr.choose_route(self))
            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

        states = self.get_state()
        done = {key: key in self.k.vehicle.get_arrived_ids()
                for key in states.keys()}
        if crash or (self.time_counter >= self.env_params.sims_per_step *
                     (self.env_params.warmup_steps + self.env_params.horizon)):
            done['__all__'] = True
        else:
            done['__all__'] = False
        infos = {key: {} for key in states.keys()}

        # compute the reward
        if self.env_params.clip_actions:
            clipped_actions = self.clip_actions(rl_actions)
            reward = self.compute_reward(clipped_actions, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)

        for rl_id in self.k.vehicle.get_arrived_rl_ids(self.env_params.sims_per_step):
            done[rl_id] = True
            reward[rl_id] = 0
            states[rl_id] = np.zeros(self.observation_space_av.shape[0])

        return states, reward, done, infos