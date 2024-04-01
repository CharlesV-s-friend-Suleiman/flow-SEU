def _get_neigh_edges(current_edge):
    """
    inner tools for getting next edges ()

    Parameters
    --------
    current_edge:str,  name of edge as 'boti_j'
    """
    next_edges = []
    edge_direction = current_edge[:-3]  # top/bot/left/right
    i, j = int(current_edge[-3]), int(current_edge[-1])  # get the index of current edge
    if edge_direction == "bot":
        next_edges.append("right" + str(i + 1) + "_" + str(j))
        next_edges.append(edge_direction+str(i)+"_"+str(j+1))

    if edge_direction == "right":
        next_edges.append(edge_direction+str(i+1)+"_"+str(j))
        next_edges.append("bot"+str(i)+"_"+str(j+1))
    return next_edges


def find_routes(edge_start, edge_end):
    """
    using DFS to find a nearly-shortest between o and d

    Parameters
    --------
    edge_start:str
    edge_end:str
    init edge_o, edge_d = "bot0_0", "bot{}_{}".format(self.row_num - 1, self.col_num)

    Returns
    ------
    route: list
    """
    def legal(edge_name): # weather the edge is in the network range, return bool
        i, j = int(edge_name[-3]), int(edge_name[-1])
        edge_direction = edge_name[:-3]
        I, J = int(edge_end[-3]), int(edge_end[-1])
        if edge_direction in ["left", "right"]:
            return i <= I and j < J
        if edge_direction in ["top", "bot"]:
            return i <= I and j <= J

# DFS
    visited = [edge_start]
    route = []
    stack = [edge_start]
    while stack is not []:
        current_edge = stack.pop()

        if current_edge == edge_end:
            route.append(current_edge)
            break

        add_route = False # if the neigh of current edge is legal, current edge can be added to route
        for next_edge in _get_neigh_edges(current_edge):
            if next_edge not in visited and legal(next_edge):
                add_route = True
                visited.append(next_edge)
                stack.append(next_edge)

        if add_route:
            route.append(current_edge)
    return route

