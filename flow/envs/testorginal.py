"""Test environment used to run simulations in the absence of autonomy."""
import gym

from flow.envs.base import Env
from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
import numpy as np


class TestEnv(Env):
    """Test environment used to run simulations in the absence of autonomy.

    Required from env_params
        None

    Optional from env_params
        reward_fn : A reward function which takes an an input the environment
        class and returns a real number.

    States
        States are an empty list.

    Actions
        No actions are provided to any RL agent.

    Rewards
        The reward is zero at every step.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        return

    def compute_reward(self, rl_actions, **kwargs):
        """See parent class."""
        if "reward_fn" in self.env_params.additional_params:
            return self.env_params.additional_params["reward_fn"](self)
        else:
            return 0

    def get_state(self, **kwargs):
        """See class definition."""
        return np.array([])


ADDITIONAL_ENV_PARAMS = {
    "max_accel" : 3,
    "max_decel" : 3,
}


class Coop1Env(Env):
    """
    env including cav and tl,this is the cav part ,
     the action space of cav is acceleration [-3, 3] m/s^-2
    """
    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))

    @property
    def observation_space(self):
        # the normalized-speed : speed/max_speed
        nor_speed = Box(
            low=0,
            high=1.2,
            shape=(3,self.initial_vehicles.num_rl_vehicles),
            dtype=np.float32)

        position = Box(
            low=-50,
            high=2000,
            shape=(3, self.initial_vehicles.num_rl_vehicles),
            dtype=np.float32)

        return Tuple((nor_speed, position))

    def _apply_rl_actions(self, rl_actions):
        # the names of all autonomous (RL) vehicles in the network
        rl_ids = self.k.vehicle.get_rl_ids()

        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.k.vehicle.apply_acceleration(rl_ids, rl_actions)
        return

    def get_state(self, **kwargs):
        # the get_ids() method is used to get the names of all rl-vehicles in the network
        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()

        # get the distance of each rl-veh with the leader and follower
        pos = self.k.vehicle.get_position(ids)
        vel = self.k.vehicle.get_speed(ids)

        # ids of f and l
        follower = self.k.vehicle.get_follower(ids)
        leader = self.k.vehicle.get_leader(ids)

        f_veh_pos = self.k.vehicle.get_position(follower)
        l_veh_pos = self.k.vehicle.get_position(leader)

        f_veh_speed = self.k.vehicle.get_speed(follower)
        l_veh_speed = self.k.vehicle.get_speed(leader)

        #  if sumo_pos error==1001 occurred, set the pos of leader and follower to -+50m of current vehicles
        #  if sumo_vel error==1001 occurred, set the v of leader and follower to max_speed of network

        for velocitylist in vel, l_veh_speed, f_veh_speed:
            for i in range(len(velocitylist)):
                if velocitylist[i] == -1001 or velocitylist[i] is None:
                    velocitylist[i] = max_speed
                velocitylist[i] /= max_speed
        for i in range(len(pos)):
            if f_veh_pos[i] == -1001:
                f_veh_pos[i] = pos[i] - 50
            if l_veh_pos[i] == -1001:
                l_veh_pos[i] = pos[i] + 50

        state = ([f_veh_speed, vel, l_veh_speed],
                 [f_veh_pos, pos, l_veh_pos])

        return state

    def compute_reward(self, rl_actions, **kwargs):
        """
        r = rv + ra, in which rv control driving fastly, ra control driving smoothly
        rv =  Sigma(v-v_max)/(k*v_max)
        ra = - (Sigma(a/a_max)^2)^0.5/k
        """

        ids = self.k.vehicle.get_rl_ids()
        max_speed = self.k.network.max_speed()
        num_rl_veh = self.k.vehicle.num_rl_vehicles

        # the reward to forward the vehicles
        average_rl_speed = sum(self.k.vehicle.get_speed(ids)) / num_rl_veh

        rv = average_rl_speed / max_speed
        if average_rl_speed - max_speed >= 5:
            rv = - 10

        reward = rv

        return reward

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
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
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
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel, smooth=False)

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

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
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

            # render a frame
            self.render()

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.sims_per_step *
                (self.env_params.warmup_steps + self.env_params.horizon)
                or crash or self.k.vehicle.num_rl_vehicles == 0)

        # compute the info for each agent
        infos = {}

        # compute the reward
        if self.env_params.clip_actions:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)

        return next_observation, reward, done, infos
