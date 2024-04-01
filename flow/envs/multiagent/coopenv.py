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


class CustomEnv(MultiEnv):
    """
    A base class for cav-tl

    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)



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