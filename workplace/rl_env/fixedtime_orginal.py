"""
rl-controlled cav & fixed time trafficlight
"""
from flow.core.params import TrafficLightParams
from flow.envs import Coop1Env
from flow.controllers import GridRecycleRouter, ExpTravelTimeRouter, RLController
from flow.core.params import SumoLaneChangeParams

import json
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.networks.traffic_light_grid import SingleIntersectionNet
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.utils import inflow_methods

# some hyper-params used in scenario

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 1,
    "max_decel": 1,
}
USE_INFLOWS = True
v_enter = 10
inner_length = 500
long_length = 700
short_length = 500
num_cars_left = 1
num_cars_right = 1
num_cars_top = 1
num_cars_bot = 1
n_columns = 4
n_rows = 3

tl_logic = TrafficLightParams(baseline=False)
phases = [{
    # N-S go through
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GGGrrrGrrrrrGGGrrrGrrrrr"
}, { # N-S go through yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GyyrrrGrrrrrGyyrrrGrrrrr"
}, {# N-S left turn
    "duration": "20",
    "minDur": "8",
    "maxDur": "25",
    "state": "GrrGGGGrrrrrGrrGGGGrrrrr"
}, {# N-S left turn yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GrryyyGrrrrrGrryyyGrrrrr"
}, { # W-E go through
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GrrrrrGGGrrrGrrrrrGGGrrr"
}, { # W-E go through yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GrrrrrGyyrrrGrrrrrGyyrrr"
}, {# W-E left turn
    "duration": "20",
    "minDur": "8",
    "maxDur": "25",
    "state": "GrrrrrGrrGGGGrrrrrGrrGGG"
}, {# W-E left turn yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GrrrrrGrryyyGrrrrrGrryyy"
}]

for center in range(n_columns*n_rows):
    tl_logic.add('center'+str(center), phases=phases, tls_type="static", programID='1')

ADDITIONAL_NET_PARAMS = {
    # dictionary of traffic light grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": n_rows,
        # number of vertical columns of edges
        "col_num": n_columns,
        # length of inner edges in the traffic light grid network
        "inner_length": inner_length,
        # length of edges where vehicles enter the network
        "short_length": short_length,
        # length of edges where vehicles exit the network
        "long_length": long_length,
        # number of cars starting at the edges heading to the top
        "cars_top": num_cars_top,
        # number of cars starting at the edges heading to the bottom
        "cars_bot": num_cars_bot,
        # number of cars starting at the edges heading to the left
        "cars_left": num_cars_left,
        # number of cars starting at the edges heading to the right
        "cars_right": num_cars_right,
    },
    # number of lanes in the horizontal edges
    "horizontal_lanes": 3,
    # number of lanes in the vertical edges
    "vertical_lanes": 3,
    # speed limit for all edges, may be represented as a float value, or a
    # dictionary with separate values for vertical and horizontal lanes
    "speed_limit": {
        "horizontal": 30,
        "vertical": 30
    },
    "traffic_lights": True
}  # the net_params

HORIZON = 3000
N_ROLLOUTS = 8
tot_cars = (num_cars_left + num_cars_right) * n_columns \
           + (num_cars_top + num_cars_bot) * n_rows

vehs = VehicleParams()
vehs.add(
    veh_id='human',
    routing_controller=(GridRecycleRouter,{}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        decel=7.5,
    ),
    num_vehicles=tot_cars-1,
    color='white',
)
vehs.add(
    veh_id='cav',
    acceleration_controller=(RLController,{}),
    routing_controller=(ExpTravelTimeRouter,{}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        decel=7.5,
    ),
    num_vehicles=1,
    color='red',
    lane_change_params=SumoLaneChangeParams(lane_change_mode="only_strategic_aggressive")
)

if USE_INFLOWS:
    initial_config, net_params = inflow_methods.get_flow_params(
        col_num=n_columns,
        row_num=n_rows,
        additional_net_params=ADDITIONAL_NET_PARAMS)
else:
    initial_config, net_params = inflow_methods.get_non_flow_params(
        enter_speed=v_enter,
        add_net_params=ADDITIONAL_NET_PARAMS)

flow_params = dict(
    exp_tag="fixedtime_cav",
    env_name=Coop1Env,
    network=SingleIntersectionNet,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        emission_path='data',
        restart_instance=True
    ),

    env=EnvParams(
        horizon=HORIZON,
        warmup_steps= HORIZON // 10,
        clip_actions=False,
        additional_params={
            "target_velocity": 25,
            "sort_vehicles": False,
            "max_accel": 1,
            "max_decel": 1,
        },
    ),

    net=net_params,
    veh=vehs,
    initial=initial_config,
    tls=tl_logic,
)


def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_gpus"] = 1  # nums of gpu
    config["num_workers"] = 9
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [3, 3]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


alg_run, gym_name, config = setup_exps()
ray.init(num_cpus=10, num_gpus=1)
trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 20,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": 50,
        },
    }
})
