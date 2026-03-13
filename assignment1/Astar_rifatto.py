import osmnx as ox
import heapq
import math
import random
import argparse
from typing import List, Optional, Tuple 
import matplotlib.pyplot as plt 

#maxspeed_handler normalize maxspeed values, converting them into consistent integers
#in order to use the same unit of measure used in the Dijkstra algorithm, we check also that time is in millesimi di ora = sistance [km] / velocity [km/h] * 1000 

def maxspeed_handler(maxspeed) -> int:
    default_maxspeed = 50 #max limit velocity in the city centre (to try 30km/h)

    #if the road does not have a maxspeed, assigh the default
    if maxspeed is None: 
        return default_maxspeed
    
    #depending on the format of maxspeed, we need to handle it differently

    #if maxspeed is a list, I clean each element, then take the minimum value
    if isinstance(maxspeed, list):
        cleaned_speeds = []
        for s in maxspeed:
            speed = str(s).lower().strip().replace(" mph", "").replace(" km/h", "").replace(" kmh", "")
            if speed == 'walk':
                cleaned_speeds.append(6) #usually walk speed is around 5-6 km/h
            else:                
                try:
                    cleaned_speeds.append(int(speed))
                except ValueError:
                    continue #if we can't convert to int, we skip this value
        if not cleaned_speeds:
            return default_maxspeed
        return min(cleaned_speeds)
    
    #if maxspeed is a string, I clean it and convert to int
    if isinstance(maxspeed, str):
        speed = maxspeed.lower().strip().replace(" mph", "").replace(" km/h", "").replace(" kmh", "")
        if speed == 'walk':
            return 6 #usually walk speed is around 5-6 km/h
        try:
            return int(speed)
        except ValueError:
            return default_maxspeed
        
def prepare_graph(city: str):
    graph = ox.graph_from_place(city, network_type="drive")
    max_speeds_found = []
    for edge in graph.edges:
        maxspeed_raw = graph.edges[edge].get("maxspeed")
        maxspeed = maxspeed_handler(maxspeed_raw)
        graph.edges[edge]["maxspeed"] = maxspeed
        graph.edges[edge]["weight"] = graph.edges[edge]["length"] / maxspeed
        graph.edges[edge]["visit_count"] = 0
        max_speeds_found.append(maxspeed)
    graph.graph["max_speed"] = max(max_speeds_found) if max_speeds_found else 50
    return graph


def reconstruct_path(G, orig: int, dest: int, algorithm: Optional[str] = None) -> Optional[List[int]]:
    for node in G.nodes:
        G.nodes[node]["size"] = 10
        G.nodes[node]["color"] = "white"
    
    for edge in G.edges:
        style_unvisited_edge(G, edge)
    
    path = [dest]
    current = dest

    while current != orig:
        previous = G.nodes[current]["previous"]
        if previous is None:
            print("No path found")
            return None
        
        style_path_edge(G, (previous, current, 0))
        if algorithm:
            key = f"{algorithm}_uses"
            G.edges[(previous, current, 0)][key] = G.edges[(previous, current, 0)].get(key, 0) + 1
        path.append(previous)
        current = previous
    
    path.reverse()
    return path

#styling functions to visualize the graph
#unvisited edge
def style_unvisited_edge(graph, edge):
    graph.edges[edge]["color"] = "white"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 0.2

#visited edge
def style_visited_edge(graph, edge):
    graph.edges[edge]["color"] = "green"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 1

#active edge (currently being explored)
def style_active_edge(graph, edge):
    graph.edges[edge]["color"] = "red"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 1

#path edge
def style_path_edge(graph, edge):
    graph.edges[edge]["color"] = "blue"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 5


#plot graph: ax is the axis on which to plot, if None a new figure is created.
#show controls whether to display the plot immediately.
#close controls whether to close the figure after plotting.
def plot_graph(graph, ax=None, show=True, close=True):
    if ax is not None:
        ax.set_facecolor("black")
        ax.figure.set_facecolor("black")
    _, current_ax = ox.plot_graph(
        graph,
        node_size=[graph.nodes[node]["size"] for node in graph.nodes],
        edge_color=[graph.edges[edge]["color"] for edge in graph.edges],
        edge_alpha=[graph.edges[edge]["alpha"] for edge in graph.edges],
        edge_linewidth=[graph.edges[edge]["linewidth"] for edge in graph.edges],
        node_color="white",
        bgcolor="black",
        ax=ax,
        show=show,
        close=close,
    )

    current_ax.set_facecolor("black")
    current_ax.figure.set_facecolor("black")
    current_ax.set_xticks([])
    current_ax.set_yticks([])
    for spine in current_ax.spines.values():
        spine.set_visible(False)

def heuristic_computation(G, node, goal, heuristic_type):
    #current node coordinates
    x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
    #goal node coordinates
    x2, y2 = G.nodes[goal]["x"], G.nodes[goal]["y"]


    #to ensure that the heuristic is consistent with the cost function used in Dijkstra, 
    #I need to divide the distance by the maximum speed found in the graph, 
    #so that the heuristic is an estimate of the time to reach the goal, rather than just the distance.
    maxspeed = float(G.graph.get("max_speed", 130)) 
    #default max speed is 130kn/h because in the heuristic computation h(n) can't never overestimate the true cost to reach the goal, otherwise A* would not be optimal.


    #x = longitudine, y=latitudine
    latitude1 = math.radians(y1)
    longitude1 = math.radians(x1)
    latitude2 = math.radians(y2)
    longitude2 = math.radians(x2)
    mean_latitude = 0.5 * (latitude1 + latitude2)
    earth_radius = 6371000.0 #[meters]

    if heuristic_type == "Manhattan":
        #Manhattan distance formula
        deltaX = earth_radius * math.cos(mean_latitude) * (longitude2 - longitude1)
        deltaY = earth_radius * (latitude2 - latitude1)
        distance = abs(deltaX) + abs(deltaY)
        return distance / maxspeed

    
    elif heuristic_type == "Euclidean":
        #euclidean distance formula
        deltaX = earth_radius * math.cos(mean_latitude) * (longitude2 - longitude1)
        deltaY = earth_radius * (latitude2 - latitude1)
        distance = math.sqrt(deltaX**2 + deltaY**2)
        return distance / maxspeed

    elif heuristic_type == "Haversine":
        #haversine distance formula
        delta_phi = latitude1 - latitude2
        delta_lambda = longitude1 - longitude2

        a = math.sin(delta_phi / 2) ** 2 + math.cos(latitude1) * math.cos(latitude2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earth_radius * c
        return distance/maxspeed
    
def astar(G, orig: int, dest: int, heuristic_type: str) -> Tuple[Optional[int], Optional[List[int]]]:
    for node in G.nodes:
        G.nodes[node]["visited"] = False #to keep track of visited nodes
        G.nodes[node]["g_score"] = float("inf") #g(n) = cost from start to current node
        G.nodes[node]["f_score"] = float("inf") #f(n) = g(n) + h(n)
        G.nodes[node]["previous"] = None #to reconstruct the path at the end
    
    G.nodes[orig]["g_score"] = 0 #cost from start to start is 0
    h_n =heuristic_computation(G, orig, dest, heuristic_type) #f(n)= 0+h(n) for the start node is just the heuristic estimate to the goal
    G.nodes[orig]["f_score"] = h_n
    print(f"Starting A* with {heuristic_type} heuristic from node {orig} to node {dest}")

    #origin and destination nodes styling
    G.nodes[orig]["size"] = 60 
    G.nodes[orig]["color"] = "yellow"
    G.nodes[dest]["size"] = 60 
    G.nodes[dest]["color"] = "orange"

    for edge in G.edges:
        style_unvisited_edge(G, edge)

    # Priority Queue
    pq = [(G.nodes[orig]["f_score"], orig)]
    step = 0

    while pq:
        _, node = heapq.heappop(pq)
        
        if node == dest:
            print(f"A* {heuristic_type} terminato in {step} iterazioni")
            path = reconstruct_path(G, orig, dest, algorithm="Astar")
            return step, path
            
        if G.nodes[node]["visited"]:
            continue

        G.nodes[node]["visited"] = True
        
        for edge in G.out_edges(node, data=True, keys=True):
            edge_key = (edge[0], edge[1], 0) #consider only the first edge between two nodes
            style_visited_edge(G, edge_key)

            neighbor = edge[1]
            weight = G.edges[edge_key]["weight"]

        
            #calculate g(n)= path cost from start to current node
            tentative_g = G.nodes[node]["g_score"] + weight
            
            #if this path to neighbor is better than any previous one, update the scores and the priority queue
            if tentative_g < G.nodes[neighbor]["g_score"]:
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["g_score"] = tentative_g
                
                #calculate h(n)
                h_n = heuristic_computation(G, neighbor, dest, heuristic_type)
                #calculate f(n)= g(n) + h(n)
                G.nodes[neighbor]["f_score"] = tentative_g + h_n
                heapq.heappush(pq, (G.nodes[neighbor]["f_score"], neighbor))

                for next in G.out_edges(neighbor):
                    style_active_edge(G, (next[0], next[1], 0))
        
        step += 1
    return None, None

    
#generate random nodes as start and end points
#optional seed parameter to ensure reproducibility of the random selection
def generate_random_nodes(G, trials: int, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    nodes = list(G.nodes)   
    edges = []
    for _ in range(trials):
        start = random.choice(nodes)
        end = random.choice(nodes)
        while end == start:
            end = random.choice(nodes)
        edges.append((start, end))
    return edges

#build collage of A* runs with different heuristics for a set of start and end nodes
def build_astar_collage(G, edges, heuristic_type: str, show: bool = True, save_path: Optional[str] = None, title_prefix: str = "Trial"):
    columns = 5
    rows = math.ceil(len(edges) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(24, max(5, rows * 5)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (start_i, end_i) in enumerate(edges):
        astar(G, start_i, end_i, heuristic_kind=heuristic_type)
        plot_graph(G, ax=axes[idx], show=False, close=False)
        axes[idx].set_title(f"{title_prefix} {idx + 1}", color="white", fontsize=10)

    for idx in range(len(edges), len(axes)):
        axes[idx].axis("off")
        axes[idx].set_facecolor("black")

    fig.patch.set_facecolor("black")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# Soluzione 1: Sposta il parametro senza default all'inizio
def run_astar(G, place_name: str, heuristic_type: str, num_trials: int = 10, seed: Optional[int] = 123):
    graph = prepare_graph(place_name)
    pairs = generate_random_nodes(G, num_trials=num_trials, seed=seed)
    iterations = []

    print("Running A*")
    print("City:", place_name)
    print("Nodes:", len(graph.nodes))
    print("Edges:", len(graph.edges))
    print("Trials:", num_trials)
    print("Heuristic:", heuristic_type)

    for i, (start, end) in enumerate(pairs, start=1):
        it, _ = astar(graph, start, end, heuristic_kind=heuristic_type)
        iterations.append(it)
        print(f"Trial {i:2d} | start={start} end={end} | Iterations={it}")

    valid_iterations = [value for value in iterations if value is not None]
    if valid_iterations:
        print("\nAverage iterations:", sum(valid_iterations) / len(valid_iterations))

    build_astar_collage(graph, pairs, heuristic_kind=heuristic_type, show=True, save_path="collage_astar.png")


if __name__ == "__main__":
    # Parametri richiesti dall'assignment
    cities = ["Turin, Piedmont, Italy", "Aosta, Aosta, Italy"]
    heuristics = ["Manhattan", "Euclidean", "Haversine"]
    trials = 10
    
    for city in cities:
        print(f"\n--- Analysis for: {city} ---")
        graph = prepare_graph(city)
        pairs = generate_random_nodes(graph, trials=10, seed=123)
        
        for h_type in heuristics:
            print(f"Executing A* with heuristic: {h_type}")
            iterations = []
            for start, end in pairs:
                it, _ = astar(graph, start, end, heuristic_type=h_type)
                if it: iterations.append(it)
            
            avg = sum(iterations) / len(iterations) if iterations else 0
            print(f"Average iterations ({h_type}): {avg:.2f}")
