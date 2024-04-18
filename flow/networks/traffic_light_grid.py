"""Contains the traffic light grid scenario class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np
from flow.utils import route_tools
from flow.utils import inflow_methods
import random
import sumolib

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


class TrafficLightGridNetwork(Network):
    """Traffic Light Grid network class.

    The traffic light grid network consists of m vertical lanes and n
    horizontal lanes, with a total of nxm intersections where the vertical
    and horizontal edges meet.

    Requires from net_params:

    * **grid_array** : dictionary of grid array data, with the following keys

      * **row_num** : number of horizontal rows of edges
      * **col_num** : number of vertical columns of edges
      * **inner_length** : length of inner edges in traffic light grid network
      * **short_length** : length of edges that vehicles start on
      * **long_length** : length of final edge in route
      * **cars_top** : number of cars starting at the edges heading to the top
      * **cars_bot** : number of cars starting at the edges heading to the
        bottom
      * **cars_left** : number of cars starting at the edges heading to the
        left
      * **cars_right** : number of cars starting at the edges heading to the
        right

    * **horizontal_lanes** : number of lanes in the horizontal edges
    * **vertical_lanes** : number of lanes in the vertical edges
    * **speed_limit** : speed limit for all edges. This may be represented as a
      float value, or a dictionary with separate values for vertical and
      horizontal lanes.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import TrafficLightGridNetwork
    >>>
    >>> network = TrafficLightGridNetwork(
    >>>     name='grid',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'grid_array': {
    >>>                 'row_num': 3,
    >>>                 'col_num': 2,
    >>>                 'inner_length': 500,
    >>>                 'short_length': 500,
    >>>                 'long_length': 500,
    >>>                 'cars_top': 20,
    >>>                 'cars_bot': 20,
    >>>                 'cars_left': 20,
    >>>                 'cars_right': 20,
    >>>             },
    >>>             'horizontal_lanes': 1,
    >>>             'vertical_lanes': 1,
    >>>             'speed_limit': {
    >>>                 'vertical': 35,
    >>>                 'horizontal': 35
    >>>             }
    >>>         },
    >>>     )
    >>> )
    """

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
        self.inner_nodes_radius = 2.9 + 3.3 * max(self.vertical_lanes,
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
        """See parent class."""

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

    # ===============================
    # ============ UTILS ============
    # ===============================

    @property
    def _inner_nodes(self):
        """Build out the inner nodes of the network.

        The inner nodes correspond to the intersections between the roads. They
        are numbered from bottom left, increasing first across the columns and
        then across the rows.

        For example, the nodes in a traffic light grid with 2 rows and 3 columns
        would be indexed as follows:

            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |

        The id of a node is then "center{index}", for instance "center0" for
        node 0, "center1" for node 1 etc.

        Returns
        -------
        list <dict>
            List of inner nodes
        """
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
        """Build out the outer nodes of the network.

        The outer nodes correspond to the extremities of the roads. There are
        two at each extremity, one where the vehicles enter the network
        (inflow) and one where the vehicles exit the network (outflow).

        Consider the following network with 2 rows and 3 columns, where the
        extremities are marked by 'x', the rows are labeled from 0 to 1 and the
        columns are labeled from 0 to 2:

                 x     x     x
                 |     |     |
        (1) x----|-----|-----|----x (*)
                 |     |     |
        (0) x----|-----|-----|----x
                 |     |     |
                 x     x     x
                (0)   (1)   (2)

        On row i, there are two nodes at the left extremity of the row, labeled
        "left_row_short{i}" and "left_row_long{i}", as well as two nodes at the
        right extremity labeled "right_row_short{i}" and "right_row_long{i}".

        On column j, there are two nodes at the bottom extremity of the column,
        labeled "bot_col_short{j}" and "bot_col_long{j}", as well as two nodes
        at the top extremity labeled "top_col_short{j}" and "top_col_long{j}".

        The "short" nodes correspond to where vehicles enter the network while
        the "long" nodes correspond to where vehicles exit the network.

        For example, at extremity (*) on row (1):
        - the id of the input node is "right_row_short1"
        - the id of the output node is "right_row_long1"

        Returns
        -------
        list <dict>
            List of outer nodes
        """
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
        """Build out the inner edges of the network.

        The inner edges are the edges joining the inner nodes to each other.

        Consider the following network with n = 2 rows and m = 3 columns,
        where the rows are indexed from 0 to 1 and the columns from 0 to 2, and
        the inner nodes are marked by 'x':

                |     |     |
        (1) ----x-----x-----x----
                |     |     |
        (0) ----x-----x-(*)-x----
                |     |     |
               (0)   (1)   (2)

        There are n * (m - 1) = 4 horizontal inner edges and (n - 1) * m = 3
        vertical inner edges, all that multiplied by two because each edge
        consists of two roads going in opposite directions traffic-wise.

        On an horizontal edge, the id of the top road is "top{i}_{j}" and the
        id of the bottom road is "bot{i}_{j}", where i is the index of the row
        where the edge is and j is the index of the column to the right of it.

        On a vertical edge, the id of the right road is "right{i}_{j}" and the
        id of the left road is "left{i}_{j}", where i is the index of the row
        above the edge and j is the index of the column where the edge is.

        For example, on edge (*) on row (0): the id of the bottom road (traffic
        going from left to right) is "bot0_2" and the id of the top road
        (traffic going from right to left) is "top0_2".

        Returns
        -------
        list <dict>
            List of inner edges
        """
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
        """Build out the outer edges of the network.

        The outer edges are the edges joining the inner nodes to the outer
        nodes.

        Consider the following network with n = 2 rows and m = 3 columns,
        where the rows are indexed from 0 to 1 and the columns from 0 to 2, the
        inner nodes are marked by 'x' and the outer nodes by 'o':

                o    o    o
                |    |    |
        (1) o---x----x----x-(*)-o
                |    |    |
        (0) o---x----x----x-----o
                |    |    |
                o    o    o
               (0)  (1)  (2)

        There are n * 2 = 4 horizontal outer edges and m * 2 = 6 vertical outer
        edges, all that multiplied by two because each edge consists of two
        roads going in opposite directions traffic-wise.

        On row i, there are four horizontal edges: the left ones labeled
        "bot{i}_0" (in) and "top{i}_0" (out) and the right ones labeled
        "bot{i}_{m}" (out) and "top{i}_{m}" (in).

        On column j, there are four vertical edges: the bottom ones labeled
        "left0_{j}" (out) and "right0_{j}" (in) and the top ones labeled
        "left{n}_{j}" (in) and "right{n}_{j}" (out).

        For example, on edge (*) on row (1): the id of the bottom road (out)
        is "bot1_3" and the id of the top road is "top1_3".

        Edges labeled by "in" are edges where vehicles enter the network while
        edges labeled by "out" are edges where vehicles exit the network.

        Returns
        -------
        list <dict>
            List of outer edges
        """
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
        """Build out connections at each inner node.

        Connections describe what happens at the intersections. Here we link
        lanes in straight lines, which means vehicles cannot turn at
        intersections, they can only continue in a straight line.
        """
        con_dict = {}

        def new_con(side, from_id, to_id, lane, signal_group):
            return [{
                "from": side + from_id,
                "to": side + to_id,
                "fromLane": str(lane),
                "toLane": str(lane),
                "signal_group": signal_group
            }]

        # build connections at each inner node
        for i in range(self.row_num):
            for j in range(self.col_num):
                node_id = "{}_{}".format(i, j)
                right_node_id = "{}_{}".format(i, j + 1)
                top_node_id = "{}_{}".format(i + 1, j)

                conn = []
                for lane in range(self.horizontal_lanes):
                    conn += new_con("bot", node_id, right_node_id, lane, 1)
                    conn += new_con("top", right_node_id, node_id, lane, 1)
                for lane in range(self.vertical_lanes):
                    conn += new_con("right", node_id, top_node_id, lane, 2)
                    conn += new_con("left", top_node_id, node_id, lane, 2)

                node_id = "center{}".format(i * self.col_num + j)
                con_dict[node_id] = conn

        return con_dict

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
        """See parent class."""
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


class SingleIntersectionNet(Network):
    """the single intersection scenario
    YX 2024
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()
                 ):
        optional = ["tl_logic"]

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
        for ids in self.vehicles.rl_ids:
            edge_o, edge_d = "bot0_0", "bot{}_{}".format(self.row_num - 1, self.col_num)
            routes[ids] = route_tools.find_routes(edge_o, edge_d)

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
                # pass through and right lanes
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
                for l in range(self.horizontal_lanes):
                    conn += [{
                            "from": "top"+right_node_id,
                            "to": "left"+node_id,
                            "fromLane": str(self.horizontal_lanes-1),
                            "toLane": str(l),
                            "signal_group": 1
                            }]
                    conn += [{
                            "from": "bot"+node_id,
                            "to": "right"+top_node_id,
                            "fromLane": str(self.horizontal_lanes-1),
                            "toLane": str(l),
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
                for l in range(self.vertical_lanes):
                    conn += [{
                        "from": "right" + node_id,
                        "to": "top" + node_id,
                        "fromLane": str(self.vertical_lanes-1),
                        "toLane": str(l),
                        "signal_group": 2
                        }]
                    conn += [{
                        "from": "left" + top_node_id,
                        "to": "bot" + right_node_id,
                        "fromLane": str(self.vertical_lanes-1),
                        "toLane": str(l),
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
                "length": self.inner_length * (0.8 + 0.4 * random.random())
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
                "length": length* (0.8 + 0.4 * random.random())
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
        and for each of them, the 4 edges that enter this node.

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

    @property
    def node_mapping_leave(self):
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

                top_edge_id = "right{}_{}".format(row + 1, col)
                bot_edge_id = "left{}_{}".format(row, col)
                right_edge_id = "bot{}_{}".format(row, col + 1)
                left_edge_id = "top{}_{}".format(row, col)

                mapping[node_id] = [left_edge_id, bot_edge_id,
                                    right_edge_id, top_edge_id]

        return sorted(mapping.items(), key=lambda x: x[0])

    def _get_incoming_edges(self):
        # Initialize a dictionary to store the incoming edges for each node
        incoming_edges = {node['id']: [] for node in self.nodes}

        # Iterate over the edges in the network
        for edge in self.edges:
            # Add the edge to the incoming edges list of the destination node
            incoming_edges[edge['to']].append(edge['id'])

        return incoming_edges

    def _get_outgoing_edges(self):
        # Initialize a dictionary to store the outgoing edges for each node
        outgoing_edges = {node['id']: [] for node in self.nodes}

        # Iterate over the edges in the network
        for edge in self.edges:
            # Add the edge to the outgoing edges list of the source node
            outgoing_edges[edge['from']].append(edge['id'])

        return outgoing_edges

    @property
    def get_edge_mappings(self):
        # the grid network has 4 incoming edges and 4 outgoing edges,
        # this should be modified if the network is changed or using real-world data
        mapping_inc, mapping_out = self._get_incoming_edges(), self._get_outgoing_edges()
        return mapping_inc, 4, mapping_out, 4

    def get_states(self):
        states_tl = {}
        for node in self._inner_nodes:
            if node['id'] not in states_tl.keys():
                states_tl[node['id']] = self.traffic_lights.get_properties()[node['id']]['id']
        return states_tl
