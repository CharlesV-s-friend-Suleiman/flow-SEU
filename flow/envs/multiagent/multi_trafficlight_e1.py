"""Multi-agent environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m traffic light grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.trafficlight_e4 import TrafficLightGrid_a_Env
from flow.envs.multiagent import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1


class MultiTrafficLightGrid_a_Env(TrafficLightGrid_a_Env, MultiEnv):
    """Multiagent shared model version of TrafficLightGridPOEnv.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(4)
        else:
            return Box(
                low=0,
                high=3.999,
                shape=(1,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """State space that is partially observed."""
        tl_box = Box(
            low=0.,
            high=1,
            shape=(4 * (1 + self.num_local_lights)+2 * 3 * 4 * self.num_observed ,),
            dtype=np.float32)
        return tl_box



    def get_state(self):
        """Observations for each traffic light agent.

        :return: dictionary which contains agent-wise observations as follows:
        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity, distance to
        intersection.
        - For the self.num_local_lights number of nearest lights (itself
        included), gives the traffic light information------NQI.
        """
        # Normalization factors
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        max_accel = max(2.6, 7.5)
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        # TODO(cathywu) refactor TrafficLightGridPOEnv with convenience
        # methods for observations, but remember to flatten for single-agent
        # Observed vehicle information
        speeds = []
        dist_to_intersec = []
        acc = []
        NQI_mean = []
        NQI_mean_0=[]
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
                local_speeds.extend(
                    [self.k.vehicle.get_speed(veh_id) / max_speed for veh_id in
                     observed_ids])
                local_dists_to_intersec.extend([(self.k.network.edge_length(
                    self.k.vehicle.get_edge(
                        veh_id)) - self.k.vehicle.get_position(
                    veh_id)) / max_dist for veh_id in observed_ids])
                local_acc.extend([self.k.vehicle.get_accel(veh_id) for veh_id in observed_ids])

                if len(observed_ids) < self.num_observed*3:
                    diff = self.num_observed*3 - len(observed_ids)
                    local_speeds.extend([0] * diff)
                    local_dists_to_intersec.extend([0] * diff)
                    local_acc.extend([0] * diff)

            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            acc.append(local_acc)

            local_NQI_mean=[]
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
        NQI_mean_0.append([0.0,0.0,0.0,0.0])
        # for i in range(len(NQI_mean) // 4):
        #     self.nqi_top[i] = NQI_mean[i * 4 + 0]
        #     self.nqi_right[i] = NQI_mean[i * 4 + 1]
        #     self.nqi_bot[i] = NQI_mean[i * 4 + 2]
        #     self.nqi_left[i] = NQI_mean[i * 4 + 3]

        self.observed_ids = all_observed_ids

        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_id_nums = [rl_id_num, self._get_relative_node(rl_id, "top"),
                             self._get_relative_node(rl_id, "bottom"),
                             self._get_relative_node(rl_id, "left"),
                             self._get_relative_node(rl_id, "right")]

            NQI_mean_1 = []
            for local_id_num in local_id_nums:
                NQI_mean_1 += NQI_mean_0[local_id_num]
            NQI_mean.append(NQI_mean_1)

        obs = {}
        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            observation = np.array(np.concatenate(
                [NQI_mean[rl_id_num], speeds[rl_id_num], dist_to_intersec[rl_id_num]]))

            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        Issues action for each traffic light agent.
        """
        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])
            if self.discrete:
                action = int(rl_action)
            else:
                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

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

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}

        if self.env_params.evaluate:
            rew = -rewards.min_delay_unscaled(self)
        else:
            rew = (-rewards.min_delay_unscaled(self) +
                   rewards.penalize_standstill(self, gain=0.2))

        # each agent receives reward normalized by number of lights
        rew /= self.num_traffic_lights

        rews = {}
        for rl_id in rl_actions.keys():
            rews[rl_id] = rew
        return rews

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        for veh_ids in self.observed_ids:
            for veh_id in veh_ids:
                self.k.vehicle.set_observed(veh_id)

    def _get_relative_node(self, agent_id, direction):
        """Yield node number of traffic light agent in a given direction.

        For example, the nodes in a traffic light grid with 2 rows and 3
        columns are indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

        See flow.networks.traffic_light_grid for more information.

        Example of function usage:
        - Seeking the "top" direction to ":center0" would return 3.
        - Seeking the "bottom" direction to ":center0" would return -1.

        Parameters
        ----------
        agent_id : str
            agent id of the form ":center#"
        direction : str
            top, bottom, left, right

        Returns
        -------
        int
            node number
        """
        ID_IDX = 1
        agent_id_num = int(agent_id.split("center")[ID_IDX])
        if direction == "top":
            node = agent_id_num + self.cols
            if node >= self.cols * self.rows:
                node = -1
        elif direction == "bottom":
            node = agent_id_num - self.cols
            if node < 0:
                node = -1
        elif direction == "left":
            if agent_id_num % self.cols == 0:
                node = -1
            else:
                node = agent_id_num - 1
        elif direction == "right":
            if agent_id_num % self.cols == self.cols - 1:
                node = -1
            else:
                node = agent_id_num + 1
        else:
            raise NotImplementedError

        return node
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
