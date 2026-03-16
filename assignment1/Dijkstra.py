#The script loads a driving network from OpenStreetMap (using osmnx), sets each edge’s travel time as its 
#weight (length / maxspeed), runs a Dijkstra shortest-path search (using heapq) between two randomly chosen nodes, 
#styles edges/nodes for visualization while the algorithm runs, reconstructs the found path, increments an 
#edge usage counter (dijkstra_uses) for the path, and finally draws a graph visualization showing visited/active/path 
#edges. The graph object G is a global osmnx MultiDiGraph.

import osmnx as ox
import random
import math
from typing import Optional
from typing import List
import heapq
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use('Agg')

def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph(G, ax=None, show=True, close=True):
    if ax is not None:
        ax.set_facecolor("black")
        ax.figure.set_facecolor("black")
    _, current_ax = ox.plot_graph(
        G,
        node_size=[G.nodes[node].get("size", 1) for node in G.nodes],
        edge_color=[G.edges[edge]["color"] for edge in G.edges],
        edge_alpha=[G.edges[edge]["alpha"] for edge in G.edges],
        edge_linewidth=[G.edges[edge]["linewidth"] for edge in G.edges],
        node_color=[G.nodes[node].get("color", "white") for node in G.nodes],
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


def dijkstra(orig, dest, plot=False) -> tuple[int, float, float]:
    # initialise node attributes
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq) #takes the node with the smallest distance from the priority queue
        if node == dest:
        
            #print("Iterations to convergence:", step)
            # Questo è il costo minimo (tempo) trovato da Dijkstra
            total_cost_time = G.nodes[dest]["distance"] 
            #print(f"total_cost_Dijkstra: {total_cost_time}") 
            #reconstruct_path to find total distance in km
            total_distance_km = reconstruct_path(orig, dest)
            #plot_graph()
            return step, total_cost_time, total_distance_km
        
        if G.nodes[node]["visited"]:
            continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node): #iterates over the outgoing edges of the current node
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            #relaxation step
            #if distance to neighbor through current node is less than previously known distance, update it and add to priority queue
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1
    # if destination wasn't reached, return final step count
    return step, None, None

#reconstructs the final path from dest to origin using the "previous" attribute of each node, styles the path edges
#increments the usage counter for each edge in the path. 
#calculates the total distance of the path in kilometers.

def reconstruct_path(G, orig: int, dest: int, algorithm: Optional[str] = None) -> Optional[List[int]]:
    for node in G.nodes:
        visited = G.nodes[node].get("visited", False)
        if node == orig:
            G.nodes[node]["size"] = 80
            G.nodes[node]["color"] = "green"
        elif node == dest:
            G.nodes[node]["size"] =80
            G.nodes[node]["color"] = "red"
        elif visited:
            G.nodes[node]["size"] = 1
            G.nodes[node]["color"] = "white"
        else:
            G.nodes[node]["size"] = 0.2
            G.nodes[node]["color"] = "white"

    for edge in G.edges:
        style_unvisited_edge(G, edge)
    
    path = [dest]
    current = dest

    distance = 0.0
    while current != orig:
        previous = G.nodes[current]["previous"]
        if previous is None:
            print("No path found")
            return None

        # get a valid edge key between the two nodes (MultiDiGraph may have multiple parallel edges)
        edge_data = G.get_edge_data(previous, current)
        if not edge_data:
            print(f"No edge data found between {previous} and {current}")
            return None
        edge_key, edge_attr = next(iter(edge_data.items()))
        edge_tuple = (previous, current, edge_key)

        style_path_edge(G, edge_tuple)
        if algorithm:
            key = f"{algorithm}_uses"
            G.edges[edge_tuple][key] = G.edges[edge_tuple].get(key, 0) + 1

        distance += edge_attr.get("length", 0) / 1000  # distance in km
        path.append(previous)
        current = previous

    path.reverse()
    return distance


def build_dijkstra_collage(G, edges, show: bool = True, save_path: Optional[str] = None, title_prefix: str = "Trial"):
    columns = 5
    rows = math.ceil(len(edges) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(24, max(5, rows * 5)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (start_i, end_i) in enumerate(edges):
        dijkstra(start_i, end_i, plot=False)
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

#metto in un vettore di stringhe le due città, una alla volta, e faccio il ciclo per entrambe
cities = ["Turin, Piedmont, Italy", "Aosta, Aosta, Italy"]

for city in cities:

    print("Loading graph for ", city)
    place_name = city
    G = ox.graph_from_place(place_name, network_type="drive")

    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        maxspeed = 50
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]
            if type(maxspeed) == list:
                #speeds = [ int(speed) for speed in maxspeed ]
                speeds = [int(speed) if speed != "walk" else 1 for speed in maxspeed]
                maxspeed = min(speeds)
            elif type(maxspeed) == str:
                if maxspeed == "walk": 
                    maxspeed = 1
                else:
                    maxspeed = maxspeed.strip(" mph")
                    maxspeed = int(maxspeed)
        G.edges[edge]["maxspeed"] = maxspeed
        # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed


    for edge in G.edges:
        G.edges[edge]["dijkstra_uses"] = 0

    os.makedirs("results", exist_ok=True)

    pairs = []
    nodes = list(G.nodes)
    for _ in range(10):
        start = random.choice(nodes)
        end = random.choice(nodes)
        while end == start:
            end = random.choice(nodes)
        pairs.append((start, end))

    print("Running Dijkstra's algorithm on ", city, "over 10 pairs of random nodes")
    iterations = []
    times = []
    distances = []
    for idx, (start, end) in enumerate(pairs, start=1):
        print(f"Running Dijkstra iteration {idx}: start={start} end={end}")
        steps, tempo, km = dijkstra(start, end)
        if tempo is not None and km is not None:
            print(f"Run {idx} | Step: {steps} | Time: {tempo:.4f} | KM Distance: {km:.2f} km")
            iterations.append(steps)
            times.append(tempo)
            distances.append(km)
        else:
            print(f"Run {idx} | Step: {steps} | Path not found")

    avg_iterations = sum(iterations) / len(iterations) if iterations else 0
    avg_time = sum(times) / len(times) if times else 0
    avg_distance = sum(distances) / len(distances) if distances else 0

    print(f"AVG_ITERATIONS_DIJKSTRA_{city}: {avg_iterations:.2f}")
    print(f"AVG_COST_DIJKSTRA_{city}: {avg_time:.6f}")
    print(f"AVG_DISTANCE_DIJKSTRA_{city}: {avg_distance:.2f}")

    save_path = f"results/{city.replace(' ', '_').replace(',', '')}_dijkstra.png"
    build_dijkstra_collage(G, pairs, save_path=save_path)
    print("Saved Dijkstra collage to", save_path)
    
