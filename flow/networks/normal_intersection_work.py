
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # dictionary of traffic light grid array data
    "grid_array": {
        # number of horizontal rows of edges
        "row_num": 3,
        # number of vertical columns of edges
        "col_num": 2,
        # length of inner edges in the traffic light grid network
        "inner_length": None,
        # length of edges where vehicles enter the network
        "short_length": None,
        # length of edges where vehicles exit the network
        "long_length": None,
        # number of cars starting at the edges heading to the top
        "cars_top": 20,
        # number of cars starting at the edges heading to the bottom
        "cars_bot": 20,
        # number of cars starting at the edges heading to the left
        "cars_left": 20,
        # number of cars starting at the edges heading to the right
        "cars_right": 20,
    },
    # number of lanes in the horizontal edges
    "horizontal_lanes": 1,
    # number of lanes in the vertical edges
    "vertical_lanes": 1,
    # speed limit for all edges, may be represented as a float value, or a
    # dictionary with separate values for vertical and horizontal lanes
    "speed_limit": {
        "horizontal": 35,
        "vertical": 35
    }
}


class TrafficLightGridNetwork2(Network):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize an n*m traffic light grid network."""
        optional = ["tl_logic"]
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params and p not in optional:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_NET_PARAMS["grid_array"].keys():
            if p not in net_params.additional_params["grid_array"]:
                raise KeyError(
                    'Grid array parameter "{}" not supplied'.format(p))

        # retrieve all additional parameters
        # refer to the ADDITIONAL_NET_PARAMS dict for more documentation
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

        # specifies whether or not there will be traffic lights at the
        # intersections (True by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", True)

        # radius of the inner nodes (ie of the intersections)
        self.inner_nodes_radius = 4 + 4 * max(self.vertical_lanes,
                                                  self.horizontal_lanes)

        # total number of edges in the network
        self.num_edges = 4 * ((self.col_num + 1) * self.row_num + self.col_num)

        # name of the network (DO NOT CHANGE)
        self.name = "BobLoblawsLawBlog"

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        return self._inner_nodes + self._outer_nodes

    def specify_edges(self, net_params):
        """See parent class."""
        return self._inner_edges + self._outer_edges

    def specify_routes(self, net_params):
        """give each edge which is not a outer-flow edge a route"""
        routes = defaultdict(list)

        # build row routes (vehicles go from left to right and vice versa)
        for i in range(self.row_num):
            for k in range(self.col_num):
                bot_id = "bot{}_{}".format(i, k)
                top_id = "top{}_{}".format(i, self.col_num - k)
                for j in range(self.col_num + 1 - k):
                    routes[bot_id] += ["bot{}_{}".format(i, j)]
                    routes[top_id] += ["top{}_{}".format(i, self.col_num - j)]

        # build column routes (vehicles go from top to bottom and vice versa)
        for j in range(self.col_num):
            for k in range(self.row_num):
                left_id = "left{}_{}".format(self.row_num - k, j)
                right_id = "right{}_{}".format(k, j)
                for i in range(self.row_num + 1 - k):
                    routes[left_id] += ["left{}_{}".format(self.row_num - i, j)]
                    routes[right_id] += ["right{}_{}".format(i, j)]
        return routes

    def specify_types(self, net_params):
        """See parent class."""
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
                })

        return nodes

    @property
    def _outer_nodes(self):

        nodes = []

        def new_node(x, y, name, i):
            return [{"id": name + str(i), "x": x, "y": y, "type": "priority"}]

        # build nodes at the extremities of columns
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

    # TODO necessary?
    def specify_edge_starts(self):
        """Specify the edge witch allow veh to init
          this function combine with next function to set the init position of veh in network

          Returns
        -------
        list of (str, float)
            list of edge names and starting positions,
            ex: [(edge0, pos0), (edge1, pos1), ...]
         """

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

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """See parent class."""
        grid_array = net_params.additional_params["grid_array"]
        row_num = grid_array["row_num"]
        col_num = grid_array["col_num"]
        cars_heading_left = grid_array["cars_left"]
        cars_heading_right = grid_array["cars_right"]
        cars_heading_top = grid_array["cars_top"]
        cars_heading_bot = grid_array["cars_bot"]

        start_pos = []

        x0 = 6  # position of the first car in the edge
        dx = 10  # distance between each car

        start_lanes = []
        for i in range(col_num):
            for j in range(row_num):
                start_pos += [("right{}_{}".format(j, i), x0 + k * dx)
                              for k in range(cars_heading_right)]
                start_pos += [("left{}_{}".format(row_num - j, i), x0 + k * dx)
                              for k in range(cars_heading_left)]
                horz_lanes = np.random.randint(low=0, high=net_params.additional_params["horizontal_lanes"],
                                               size=cars_heading_left + cars_heading_right).tolist()
                start_lanes += horz_lanes

        for i in range(row_num):
            for j in range(col_num):
                start_pos += [("top{}_{}".format(i, col_num - j), x0 + k * dx)
                              for k in range(cars_heading_top)]
                start_pos += [("bot{}_{}".format(i, j), x0 + k * dx)
                              for k in range(cars_heading_bot)]
                vert_lanes = np.random.randint(low=0, high=net_params.additional_params["vertical_lanes"],
                                               size=cars_heading_left + cars_heading_right).tolist()
                start_lanes += vert_lanes

        return start_pos, start_lanes

    @property
    def node_mapping(self):

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


