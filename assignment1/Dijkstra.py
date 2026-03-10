#The script loads a driving network from OpenStreetMap (using osmnx), sets each edge’s travel time as its 
#weight (length / maxspeed), runs a Dijkstra shortest-path search (using heapq) between two randomly chosen nodes, 
#styles edges/nodes for visualization while the algorithm runs, reconstructs the found path, increments an 
#edge usage counter (dijkstra_uses) for the path, and finally draws a graph visualization showing visited/active/path 
#edges. The graph object G is a global osmnx MultiDiGraph.

import osmnx as ox
import random
import heapq

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

def plot_graph():
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black"
    )

def dijkstra(orig, dest, plot=False):
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
            print("Iterations to convergence:", step)
            #plot_graph()
            return step
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
    return step

#reconstructs the final path from dest to origin using the "previous" attribute of each node, styles the path edges
#increments the usage counter for each edge in the path. 
#calculates the total distance of the path in kilometers.
def reconstruct_path(orig, dest, plot=False, algorithm=None):
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    speeds = []
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000


#metto in un vettore di stringhe le due città, una alla volta, e faccio il ciclo per entrambe
cities = ["Turin, Piedmont, Italy", "Aosta, Aosta, Italy"]

for city in cities:

    print("Loading graph for ", city)
    place_name = city
    G = ox.graph_from_place(place_name, network_type="drive")

    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        maxspeed = 40
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

    convergenceIterations = []

    print("Running Dijkstra's algorithm on ", city, "over 10 pairs of random nodes")
    for i in range(10):
        print("Running Dijkstra, iteration ", i+1)

        start = random.choice(list(G.nodes))
        print("Start node: ", start)
        end = random.choice(list(G.nodes))
        print("End node: ", end)

        print("Nodes: ", len(G.nodes))
        print("Edges: ", len(G.edges))
        iterations = dijkstra(start, end)
        convergenceIterations.append(iterations)
        print( "Done for the", i+1, "iteration, reconstructing path and plotting graph" )

        reconstruct_path(start, end, algorithm="dijkstra", plot=True)
        plot_graph()

    #calculate average iterations over 10 pairs of randomly selected points
    avg_iterations = sum(convergenceIterations) / len(convergenceIterations)
    print("Done running Dijkstra for ", city)
    print("Average iterations to convergence over 10 runs:", avg_iterations)
    