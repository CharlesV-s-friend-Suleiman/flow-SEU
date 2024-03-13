"""Traffic Light Grid example."""
import json
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController
from flow.envs.trafficlight_e4 import TrafficLightGrid_a_Env
from flow.networks.traffic_light_grid import SingleIntersectionNet
from flow.controllers import GridRecycleRouter,GridRouter
from numba import cuda
import matplotlib.pyplot as plt


# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 5
# number of parallel workers
N_CPUS = 10
# N_GPUS = 4
# set to True if you would like to run the experiment with inflows of vehicles
# from the edges, and False otherwise
USE_INFLOWS = True


def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the traffic light grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_inflow_params(col_num, row_num, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid
    additional_net_params : dict
        network-specific parameters that are unique to the traffic light grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            probability=0.20,
            depart_lane='free',   #free与random
            depart_speed=10)

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the traffic light grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net = NetParams(additional_params=add_net_params)

    return initial, net


v_enter = 10
inner_length = 300
long_length = 700
short_length = 500
num_cars_left = 1
num_cars_right = 1
num_cars_top = 1
num_cars_bot = 1
n_columns = 1
n_rows = 1

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

tot_cars = ((num_cars_left + num_cars_right) * n_columns
           + (num_cars_top + num_cars_bot) * n_rows)

ADDITIONAL_ENV_PARAMS = {
        'target_velocity': 30,
        'switch_time': 3.0,
        'num_observed': 4,
        'discrete': True,
        'tl_type': 'controlled'
    }


vehicles = VehicleParams()
vehicles.add(
    veh_id='idm',
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_Gap=2.5,
        decel=3.5,  # avoid collisions at emergency stops
        speed_mode="all_checks",
    ),
    routing_controller=(GridRecycleRouter, {}),
    num_vehicles=tot_cars)

# collect the initialization and network-specific parameters based on the
# choice to use inflows or not
if USE_INFLOWS:
    initial_config, net_params = get_inflow_params(
        col_num=n_columns,
        row_num=n_rows,
        additional_net_params=ADDITIONAL_NET_PARAMS)
else:
    initial_config, net_params = get_non_flow_params(
        enter_speed=v_enter,
        add_net_params=ADDITIONAL_NET_PARAMS)

flow_params = dict(
    # name of the experiment
    exp_tag='traffic_light_grid',

    # name of the flow environment the experiment is running on
    env_name=TrafficLightGrid_a_Env,

    # name of the network class the experiment is running on
    network=SingleIntersectionNet,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=1,
        render=False,
        restart_instance=True,
        print_warnings=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=0,
        evaluate=False,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component). This is
    # filled in by the setup_exps method below.
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig). This is filled in by the
    # setup_exps method below.
    initial=initial_config,
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
    config["num_gpus"] = 1
    config["num_workers"] = 10
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [3, 3]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["vf_clip_param"] = 1000
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
ray.init(num_cpus=N_CPUS + 1,num_gpus=1)
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
            "training_iteration": 500,
        },
    }
})

