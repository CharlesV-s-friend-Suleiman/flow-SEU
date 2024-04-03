"""
Environment for implement of cooperation of trafficlight and autonomous vehicles via reinforcement-learning method
"""
from random import choice
from flow.core import rewards

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
        self.mapping_inc, self.num_local_edges_max, self.mapping_out, self.num_out_edges_max = network.node_mapping
        self.lanes_related = []
        for each in self.mapping_inc.values():
            self.lanes_related.extend(each)
        for each in self.mapping_out.values():
            self.lanes_related.extend(each)
        # after 2 loops, self.lanes_related contains all the lanes around the intersections

        # traffic light
        self.states_tl = network.get_states()
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
        """State space that is partially observed."""
        tl_box = Box(
            low=0.,
            high=1,
            shape=(4 * (1 + self.num_local_lights)
                   + 2 * 3 * 4 * self.num_observed,),
            dtype=np.float32)
        return tl_box

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

            veh_num_per_in = [veh_num_per_edge[each] for each in local_edges]
            veh_num_per_out = [veh_num_per_edge[each] for each in local_edges_out]

            for cav_id in incoming_tl.keys():
                if self.full_name_edge_lane(cav_id) in local_edges:
                    incoming_tl[cav_id] = tl_id_num  # get the id of the approaching TL
            states = self.states_tl[tl_id]
            now_state = self.k.traffic_light.get_state(tl_id)
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
                rv -= - 10
            reward[rl_id] = rv

        return reward

    def _apply_rl_actions(self, rl_actions):
        for rl_id, rl_action in rl_actions.items():
            # light
            if rl_id in self.mapping_inc.keys():
                tl_id_num = list(self.mapping_inc.keys()).index(rl_id)
                action = rl_action > 0.0

                states = self.states_tl[rl_id]
                now_state = self.k.traffic_light.get_state(rl_id)
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

