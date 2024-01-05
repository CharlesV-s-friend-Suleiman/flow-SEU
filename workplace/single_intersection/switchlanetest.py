from flow.core.params import SumoParams, EnvParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.core.experiment import Experiment
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers import GridRecycleRouter
from flow.networks.traffic_light_grid import SingleIntersectionNet
'''tools from workspace'''
from flow.utils import inflow_methods

# some hyper-params used in scenario
USE_INFLOWS = True

v_enter = 10
inner_length = 500
long_length = 700
short_length = 500
num_cars_left = 1
num_cars_right = 1
num_cars_top = 1
num_cars_bot = 1
n_columns = 5
n_rows = 2

tl_logic = TrafficLightParams(baseline=False)
phases = [{
    # N-S go through
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GGGrGrrrGGGrGrrr"
}, { # N-S go through yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GyyrGrrrGyyrGrrr"
}, {# N-S left turn
    "duration": "20",
    "minDur": "8",
    "maxDur": "25",
    "state": "GrrGGrrrGrrGGrrr"
}, {# N-S left turn yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GrryGrrrGrryGrrr"
}, { # W-E go through
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    "state": "GrrrGGGrGrrrGGGr"
}, { # W-E go through yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GrrrGyyrGrrrGyyr"
}, {# W-E left turn
    "duration": "20",
    "minDur": "8",
    "maxDur": "25",
    "state": "GrrrGrrGGrrrGrrG"
}, {# W-E left turn yellow
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    "state": "GrrrGrryGrrrGrry"
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
        "horizontal": 35,
        "vertical": 35
    },
    "traffic_lights": True
}  # the net_params

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
    #lane_change_controller=(ImmediateLaneChanger,{}),
    num_vehicles=tot_cars-1,
    color='white',
)
vehs.add(
    veh_id='test',
    routing_controller=(GridRecycleRouter,{}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        decel=7.5,
    ),
    #lane_change_controller=(ImmediateLaneChanger,{}),
    num_vehicles=1,
    color='red',
)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)


"""params using in simulation"""
sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')
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
    exp_tag='test_network',
    env_name=AccelEnv,
    network=SingleIntersectionNet,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehs,
    initial=initial_config,
    tls=tl_logic
)


'''exp run'''
flow_params['env'].horizon = 3000
exp = Experiment(flow_params)
_ = exp.run(1, convert_to_csv=True)