"""Contains a list of custom routing controllers."""
import random
import numpy as np

from flow.controllers.base_routing_controller import BaseRouter


class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed ring.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class.

        Adopt one of the current edge's routes if about to leave the network.
        """
        edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif edge == current_route[-1]:
            # choose one of the available routes based on the fraction of times
            # the given route can be chosen
            num_routes = len(env.available_routes[edge])
            frac = [val[1] for val in env.available_routes[edge]]
            route_id = np.random.choice(
                [i for i in range(num_routes)], size=1, p=frac)[0]

            # pass the chosen route
            return env.available_routes[edge][route_id][0]
        else:
            return None


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity network.

    This class allows the vehicle to pick a random route at junctions.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.k.vehicle
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.k.network.next_edge(veh_edge,
                                                vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge:
            random_route = random.randint(0, len(veh_next_edge) - 1)
            while veh_next_edge[0][0][0] == not_an_edge:
                veh_next_edge = env.k.network.next_edge(
                    veh_next_edge[random_route][0],
                    veh_next_edge[random_route][1])
            next_route = [veh_edge, veh_next_edge[0][0]]
        else:
            next_route = None

        if veh_edge in ['e_37', 'e_51']:
            next_route = [veh_edge, 'e_29_u', 'e_21']

        return next_route


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle in a traffic light grid environment.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        if len(env.k.vehicle.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.k.vehicle.get_edge(self.veh_id) == \
                env.k.vehicle.get_route(self.veh_id)[-1]:
            return [env.k.vehicle.get_edge(self.veh_id)]
        else:
            return None


class GridRecycleRouter(BaseRouter):
    """A router used to re-route a vehicle in a traffic light grid environment in ring.
    written by YangX 2023.04.13

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class"""

        def get_next(c_e, method):
            """A function used to get the next edge due to current position and lane """
            edge_direction = c_e[:-3] # top/bot/left/right
            i,j = int(c_e[-3]), int(c_e[-1]) # get the index of current edge
            n_e = None
            if method=="r":
                if edge_direction=="bot":n_e="left"+str(i)+"_"+str(j)
                elif edge_direction=="right":n_e="bot"+str(i)+"_"+str(j+1)
                elif edge_direction=="top":n_e="right"+str(i+1)+"_"+str(j-1)
                else: n_e="top"+str(i-1)+"_"+str(j)

            elif method=="l":
                if edge_direction=="bot":n_e="right"+str(i+1)+"_"+str(j)
                elif edge_direction=="right":n_e="top"+str(i)+"_"+str(j)
                elif edge_direction=="top":n_e="left"+str(i)+"_"+str(j-1)
                else:n_e="bot"+str(i-1)+"_"+str(j+1)
            return n_e

        def finish_the_travel(veh_id):
            """A function to judge if the veh is finish its travel and move out the network"""
            is_finish = False

            c_e = env.k.vehicle.get_edge(veh_id)
            edge_direction = c_e[:-3]  # top/bot/left/right

            n, m = env.network.row_num, env.network.col_num  # get the network size
            i, j = int(c_e[-3]), int(c_e[-1])

            if edge_direction == "left" and i == 0:
                is_finish = True
            elif edge_direction == "right" and i == n:
                is_finish = True
            elif edge_direction == "top" and j == 0:
                is_finish = True
            elif edge_direction == "bot" and j == m:
                is_finish = True
            else: pass
            return is_finish

        if len(env.k.vehicle.get_route(self.veh_id)) == 0:
            return None
        elif env.k.vehicle.get_edge(self.veh_id) in env.k.vehicle.get_route(self.veh_id)\
                and not finish_the_travel(self.veh_id):

            current_edge = env.k.vehicle.get_edge(self.veh_id)
            if env.k.vehicle.get_lane(self.veh_id) == 0:  # right-turn routes
                next_edge = get_next(current_edge,method="r")
                return [current_edge,next_edge]
            elif env.k.vehicle.get_lane(self.veh_id) == 2:  # left-turn routes
                next_edge = get_next(current_edge, method="l")
                return [current_edge,next_edge]
            else: # pass-through routes
                if env.k.vehicle.get_edge(self.veh_id) == env.k.vehicle.get_route(self.veh_id)[-1]:
                    return [current_edge]
                else: return None

        else:
            return None


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route


class I210Router(ContinuousRouter):
    """Assists in choosing routes in select cases for the I-210 sub-network.

    Extension to the Continuous Router.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.k.vehicle.get_edge(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)

        # vehicles on these edges in lanes 4 and 5 are not going to be able to
        # make it out in time
        if edge == "119257908#1-AddedOffRampEdge" and lane in [5, 4, 3]:
            new_route = env.available_routes[
                "119257908#1-AddedOffRampEdge"][0][0]
        else:
            new_route = super().choose_route(env)

        return new_route
