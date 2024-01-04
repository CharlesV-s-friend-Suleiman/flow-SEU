from flow.networks import Network
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.core.experiment import Experiment
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers import GridRouter, GridRecycleRouter

'''tools from workspace'''
from utils import inflow_methods, route_tools

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


class SingleIntersectionNet(Network):
    """the single intersection scenario
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=tl_logic
                 ):
        optional = ["tl_logic"]

        """to test the params inflowed"""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params and p not in optional:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_NET_PARAMS["grid_array"].keys():
            if p not in net_params.additional_params["grid_array"]:
                raise KeyError(
                    'Grid array parameter "{}" not supplied'.format(p))

        """set the num of lanes and speed limit and all"""
        self.vertical_lanes = net_params.additional_params["vertical_lanes"]
        self.horizontal_lanes = net_params.additional_params[
            "horizontal_lanes"]
        self.speed_limit = net_params.additional_params["speed_limit"]
        if not isinstance(self.speed_limit, dict):
            self.speed_limit = {
                "horizontal": self.speed_limit,
                "vertical": self.speed_limit
            }

        self.grid_array = net_params.additional_params["grid_array"]
        self.row_num = self.grid_array["row_num"]
        self.col_num = self.grid_array["col_num"]
        self.inner_length = self.grid_array["inner_length"]
        self.short_length = self.grid_array["short_length"]
        self.long_length = self.grid_array["long_length"]
        self.cars_heading_top = self.grid_array["cars_top"]
        self.cars_heading_bot = self.grid_array["cars_bot"]
        self.cars_heading_left = self.grid_array["cars_left"]
        self.cars_heading_right = self.grid_array["cars_right"]

        """set whether the intersection has a light, default=true"""
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", True)

        """set the inner of intersection(for safety)"""
        self.inner_nodes_radius = 2.9 + 3.3 * max(self.vertical_lanes,
                                                  self.horizontal_lanes)

        self.num_edges = 4 * ((self.col_num + 1) * self.row_num + self.col_num)

        # name of the network (DO NOT CHANGE)
        self.name = name

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        return self._inner_nodes + self._outer_nodes

    def specify_edges(self, net_params):
        """See parent class."""
        return self._inner_edges + self._outer_edges

    def specify_routes(self, net_params):
        """The normal direct routes and a special route generated for the CAV."""
        routes = defaultdict(list)

        # build row routes (vehicles go from left to right and vice versa)
        for i in range(self.row_num):
            bot_id = "bot{}_0".format(i)
            top_id = "top{}_{}".format(i, self.col_num)
            for j in range(self.col_num + 1):
                routes[bot_id] += ["bot{}_{}".format(i, j)]
                routes[top_id] += ["top{}_{}".format(i, self.col_num - j)]

        # build column routes (vehicles go from top to bottom and vice versa)
        for j in range(self.col_num):
            left_id = "left{}_{}".format(self.row_num, j)
            right_id = "right0_{}".format(j)
            for i in range(self.row_num + 1):
                routes[left_id] += ["left{}_{}".format(self.row_num - i, j)]
                routes[right_id] += ["right{}_{}".format(i, j)]

        # build a per-generated route for the CAV
        special_id = vehs.ids[-1]
        edge_o, edge_d = "bot0_0", "bot{}_{}".format(self.row_num - 1, self.col_num)
        routes[special_id] = route_tools.find_routes(edge_o, edge_d)

        return routes

    def specify_types(self, net_params):
        types = [{
            "id": "horizontal",
            "numLanes": self.horizontal_lanes,
            "speed": self.speed_limit["horizontal"]
        }, {
            "id": "vertical",
            "numLanes": self.vertical_lanes,
            "speed": self.speed_limit["vertical"]
        }]

        return types

    def specify_connections(self, net_params):
        con_dict = {}

        def new_con(side, from_id, to_id, lane, signal_group):
            """this method only for go through connections """
            return [{
                "from": side + from_id,
                "to": side + to_id,
                "fromLane": str(lane),
                "toLane": str(lane),
                "signal_group": signal_group
            }]

        # build connections at each inner node
        # build go through connection
        for i in range(self.row_num):
            for j in range(self.col_num):
                node_id = "{}_{}".format(i, j)
                right_node_id = "{}_{}".format(i, j + 1)
                top_node_id = "{}_{}".format(i + 1, j)


                conn = []

                for lane in range(self.horizontal_lanes-1):
                    conn += new_con("bot", node_id, right_node_id, lane, 1) # horizontal ->
                    conn += new_con("top", right_node_id, node_id, lane, 1) # horizontal <-
                    if lane == 0:
                        # right lanes
                        conn += [{
                                    "from": "top"+right_node_id,
                                    "to": "right"+top_node_id,
                                    "fromLane": str(lane),
                                    "toLane": str(lane),
                                    "signal_group": 1
                                    }]
                        conn += [{
                                    "from": "bot"+node_id,
                                    "to": "left"+node_id,
                                    "fromLane": str(lane),
                                    "toLane": str(lane),
                                    "signal_group": 1
                                    }]
                # left lanes
                conn += [{
                            "from": "top"+right_node_id,
                            "to": "left"+node_id,
                            "fromLane": str(self.horizontal_lanes-1),
                            "toLane": str(self.horizontal_lanes-1),
                            "signal_group": 1
                            }]
                conn += [{
                            "from": "bot"+node_id,
                            "to": "right"+top_node_id,
                            "fromLane": str(self.horizontal_lanes-1),
                            "toLane": str(self.horizontal_lanes-1),
                            "signal_group": 1
                            }]

                for lane in range(self.vertical_lanes-1):
                    conn += new_con("right", node_id, top_node_id, lane, 2)  # vectical /|\
                    conn += new_con("left", top_node_id, node_id, lane, 2)  # vectical \|/
                    if lane == 0:

                        conn += [{
                            "from": "right" + node_id,
                            "to": "bot" + right_node_id,
                            "fromLane": str(lane),
                            "toLane": str(lane),
                            "signal_group": 2
                        }]
                        conn += [{
                            "from": "left" + top_node_id,
                            "to": "top" + node_id,
                            "fromLane": str(lane),
                            "toLane": str(lane),
                            "signal_group": 2
                        }]

                # left lanes
                conn += [{
                    "from": "right" + node_id,
                    "to": "top" + node_id,
                    "fromLane": str(self.vertical_lanes-1),
                    "toLane": str(self.vertical_lanes-1),
                    "signal_group": 2
                }]
                conn += [{
                    "from": "left" + top_node_id,
                    "to": "bot" + right_node_id,
                    "fromLane": str(self.vertical_lanes-1),
                    "toLane": str(self.vertical_lanes-1),
                    "signal_group": 2
                }]

                node_id = "center{}".format(i * self.col_num + j)
                con_dict[node_id] = conn

        return con_dict

    @property
    def _inner_nodes(self):
        node_type = "traffic_light" if self.use_traffic_lights else "priority"
        nodes = []
        for row in range(self.row_num):
            for col in range(self.col_num):
                nodes.append({
                    "id": "center{}".format(row * self.col_num + col),
                    "x": col * self.inner_length,
                    "y": row * self.inner_length,
                    "type": node_type,
                    "radius": self.inner_nodes_radius
                }) # add node to nodes
        return nodes

    @property
    def _outer_nodes(self):
        nodes = []

        def new_node(x, y, name, i): # a simple generate string func
                return [{"id": name + str(i), "x": x, "y": y, "type": "priority"}]

        for col in range(self.col_num):
            x = col * self.inner_length
            y = (self.row_num - 1) * self.inner_length
            nodes += new_node(x, - self.short_length, "bot_col_short", col)
            nodes += new_node(x, - self.long_length, "bot_col_long", col)
            nodes += new_node(x, y + self.short_length, "top_col_short", col)
            nodes += new_node(x, y + self.long_length, "top_col_long", col)

        # build nodes at the extremities of rows
        for row in range(self.row_num):
            x = (self.col_num - 1) * self.inner_length
            y = row * self.inner_length
            nodes += new_node(- self.short_length, y, "left_row_short", row)
            nodes += new_node(- self.long_length, y, "left_row_long", row)
            nodes += new_node(x + self.short_length, y, "right_row_short", row)
            nodes += new_node(x + self.long_length, y, "right_row_long", row)

        return nodes

    @property
    def _inner_edges(self):
        edges = []

        def new_edge(index, from_node, to_node, orientation, lane):
            return [{
                "id": lane + index,
                "type": orientation,
                "priority": 78,
                "from": "center" + str(from_node),
                "to": "center" + str(to_node),
                "length": self.inner_length
            }]

        # Build the horizontal inner edges
        for i in range(self.row_num):
            for j in range(self.col_num - 1):
                node_index = i * self.col_num + j
                index = "{}_{}".format(i, j + 1)
                edges += new_edge(index, node_index + 1, node_index,
                                  "horizontal", "top")
                edges += new_edge(index, node_index, node_index + 1,
                                  "horizontal", "bot")

        # Build the vertical inner edges
        for i in range(self.row_num - 1):
            for j in range(self.col_num):
                node_index = i * self.col_num + j
                index = "{}_{}".format(i + 1, j)
                edges += new_edge(index, node_index, node_index + self.col_num,
                                  "vertical", "right")
                edges += new_edge(index, node_index + self.col_num, node_index,
                                  "vertical", "left")
        return edges

    @property
    def _outer_edges(self):
        edges = []

        def new_edge(index, from_node, to_node, orientation, length):
            return [{
                "id": index,
                "type": {"v": "vertical", "h": "horizontal"}[orientation],
                "priority": 78,
                "from": from_node,
                "to": to_node,
                "length": length
            }]

        for i in range(self.col_num):
            # bottom edges
            id1 = "right0_{}".format(i)
            id2 = "left0_{}".format(i)
            node1 = "bot_col_short{}".format(i)
            node2 = "center{}".format(i)
            node3 = "bot_col_long{}".format(i)
            edges += new_edge(id1, node1, node2, "v", self.short_length)
            edges += new_edge(id2, node2, node3, "v", self.long_length)

            # top edges
            id1 = "left{}_{}".format(self.row_num, i)
            id2 = "right{}_{}".format(self.row_num, i)
            node1 = "top_col_short{}".format(i)
            node2 = "center{}".format((self.row_num - 1) * self.col_num + i)
            node3 = "top_col_long{}".format(i)
            edges += new_edge(id1, node1, node2, "v", self.short_length)
            edges += new_edge(id2, node2, node3, "v", self.long_length)

        for j in range(self.row_num):
            # left edges
            id1 = "bot{}_0".format(j)
            id2 = "top{}_0".format(j)
            node1 = "left_row_short{}".format(j)
            node2 = "center{}".format(j * self.col_num)
            node3 = "left_row_long{}".format(j)
            edges += new_edge(id1, node1, node2, "h", self.short_length)
            edges += new_edge(id2, node2, node3, "h", self.long_length)

            # right edges
            id1 = "top{}_{}".format(j, self.col_num)
            id2 = "bot{}_{}".format(j, self.col_num)
            node1 = "right_row_short{}".format(j)
            node2 = "center{}".format((j + 1) * self.col_num - 1)
            node3 = "right_row_long{}".format(j)
            edges += new_edge(id1, node1, node2, "h", self.short_length)
            edges += new_edge(id2, node2, node3, "h", self.long_length)

        return edges

    """something didnt mean"""

    # TODO necessary?
    def specify_edge_starts(self):
        """See parent class."""
        edgestarts = []
        for i in range(self.col_num + 1):
            for j in range(self.row_num + 1):
                index = "{}_{}".format(j, i)
                if i != self.col_num:
                    edgestarts += [("left" + index, 0 + i * 50 + j * 5000),
                                   ("right" + index, 10 + i * 50 + j * 5000)]
                if j != self.row_num:
                    edgestarts += [("top" + index, 15 + i * 50 + j * 5000),
                                   ("bot" + index, 20 + i * 50 + j * 5000)]

        return edgestarts

    # TODO necessary?
    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """ Returns
        -------
        list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        list of int
            list of start lanes"""

        grid_array = net_params.additional_params["grid_array"]
        row_num = grid_array["row_num"]
        col_num = grid_array["col_num"]
        cars_heading_left = grid_array["cars_left"]
        cars_heading_right = grid_array["cars_right"]
        cars_heading_top = grid_array["cars_top"]
        cars_heading_bot = grid_array["cars_bot"]

        start_pos = []

        x0 = 6  # position of the first car
        dx = 10  # distance between each car

        start_lanes = []
        for i in range(col_num):
            start_pos += [("right0_{}".format(i), x0 + k * dx)
                          for k in range(cars_heading_right)]
            start_pos += [("left{}_{}".format(row_num, i), x0 + k * dx)
                          for k in range(cars_heading_left)]
            horz_lanes = np.random.randint(low=0, high=net_params.additional_params["horizontal_lanes"],
                                           size=cars_heading_left + cars_heading_right).tolist()
            start_lanes += horz_lanes

        for i in range(row_num):
            start_pos += [("top{}_{}".format(i, col_num), x0 + k * dx)
                          for k in range(cars_heading_top)]
            start_pos += [("bot{}_0".format(i), x0 + k * dx)
                          for k in range(cars_heading_bot)]
            vert_lanes = np.random.randint(low=0, high=net_params.additional_params["vertical_lanes"],
                                           size=cars_heading_left + cars_heading_right).tolist()
            start_lanes += vert_lanes
        # temp: add one pre-defined car from left of the network, del or complete it in future
        start_pos[-1] = ('bot0_0', x0 + cars_heading_bot * dx)
        start_lanes[-1] = 1
        return start_pos, start_lanes

    @property
    def node_mapping(self):
        """Map nodes to edges.

        Returns a list of pairs (node, connected edges) of all inner nodes
        and for each of them, the 4 edges that leave this node.

        The nodes are listed in alphabetical order, and within that, edges are
        listed in order: [bot, right, top, left].
        """
        mapping = {}

        for row in range(self.row_num):
            for col in range(self.col_num):
                node_id = "center{}".format(row * self.col_num + col)

                top_edge_id = "left{}_{}".format(row + 1, col)
                bot_edge_id = "right{}_{}".format(row, col)
                right_edge_id = "top{}_{}".format(row, col + 1)
                left_edge_id = "bot{}_{}".format(row, col)

                mapping[node_id] = [left_edge_id, bot_edge_id,
                                    right_edge_id, top_edge_id]

        return sorted(mapping.items(), key=lambda x: x[0])


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