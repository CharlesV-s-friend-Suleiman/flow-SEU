"""Contains a list of custom lane change controllers."""

from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController


class SimLaneChangeController(BaseLaneChangeController):
    """A controller used to enforce sumo lane-change dynamics on a vehicle.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return None


class StaticLaneChanger(BaseLaneChangeController):
    """A lane-changing model used to keep a vehicle in the same lane.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return 0


class CAVLaneChanger(BaseLaneChangeController):
    """
    act the lane change simply to match the route
    """
    def get_lane_change_action(self, env):
        # utils
        def get_to_node(edge_id:str, edgeSet:list)->str:
            _to_node = None
            for edge in edgeSet:
                if edge["id"] == edge_id:
                    _to_node = edge["to"]
                    break
            return _to_node

        def get_turn_direction(c,n):
            if c == 'top':
                if n == 'right': return "r", 'horizontal'
                if n == 'left' : return "l", 'horizontal'
            if c == 'bot':
                if n == 'right': return "l", 'horizontal'
                if n == 'left' : return "r", 'horizontal'

            if c == 'left':
                if n == 'bot':return "l", 'vertical'
                if n == 'top': return "r", 'vertical'
            if c == 'right':
                if n == 'bot':return "r", 'vertical'
                if n == 'top': return "l", 'vertical'

            return 't', 'all'

        # get the information of c_e and n_e and connection of link node
        vehicles = env.k.vehicle
        veh_id = self.veh_id

        veh_route = vehicles.get_route(veh_id)
        current_edge_id = vehicles.get_edge(veh_id)  # str

        if current_edge_id not in env.network.edges:
            return 0  # inner intersection

        next_edge_id = veh_route[veh_route.index(current_edge_id) + 1]  # str, next edge
        to_node = get_to_node(current_edge_id, env.network.edges)
        connection = env.network.connections[to_node]  # connection info<from,to,fromLane,toLane,signal>>
        num_lanes = {'vertical': env.network.net_params.additional_params["vertical_lanes"]
                    , 'horizontal': env.network.net_params.additional_params["horizontal_lanes"]
                     }

        # judge request of lane change
        c_direction, n_direction = current_edge_id[:-3], next_edge_id[:-3]
        turn_direction, lane_type = get_turn_direction(c_direction, n_direction)

        if c_direction == n_direction:
            return 0  # pass through
        elif turn_direction == 'l' and \
                num_lanes[lane_type]-1 != vehicles.get_lane(veh_id):  # should left turn and not in left turn lanes
            return 1






