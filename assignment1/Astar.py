#implement A* algorithm n three diMerent variants corresponding to three diMerent heuristic functions:
# a. Manhattan distance: ℎ(𝑛) = |𝑥1− 𝑥2| + |𝑦1− 𝑦2| 
# b. Euclidean distance: ℎ(𝑛) = √((𝑥1− 𝑥2)² + (𝑦1− 𝑦2)²) 
# c. Haversine distance. It computes the great-circle distance between two points on a sphere given their latitude (∅) and longitude (𝜆) in radians.


import osmnx as ox
import heapq
import math
import random

#define first heuristic function: Manhattan distance
def manhattan_heuristic(node, goal, maxspeed):
    #current node coordinates
    x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
    #goal node coordinates
    x2, y2 = G.nodes[goal]["x"], G.nodes[goal]["y"]

    #mahattan distance formula
    distance = abs(x1 - x2) + abs(y1 - y2)
    return distance/maxspeed

#implement A* algorithm
def astar(orig, dest):
    for node in G.nodes:
        G.nodes[node]["visited"] = False #to keep track of visited nodes
        G.nodes[node]["g_score"] = float("inf") #g(n) = cost from start to current node
        G.nodes[node]["f_score"] = float("inf") #f(n) = g(n) + h(n)
        G.nodes[node]["previous"] = None #to reconstruct the path at the end

    G.nodes[orig]["g_score"] = 0 #cost from start to start is 0
    G.nodes[orig]["f_score"] = manhattan_heuristic(orig, dest, G.nodes[orig]["maxspeed"]) #f(n)= 0+h(n) for the start node is just the heuristic estimate to the goal
    
    # Priority Queue 
    pq = [(G.nodes[orig]["f_score"], orig)]
    step = 0
    
    while pq:
        current_f, node = heapq.heappop(pq)
        
        if node == dest:
            print(f"A* Manhattan terminato in {step} iterazioni")
            return step
            
        if G.nodes[node]["visited"]:
            continue
        G.nodes[node]["visited"] = True
        
        for u, v, k, data in G.out_edges(node, data=True, keys=True):
            neighbor = v
            weight = data["weight"]
            
            #calculate g(n)= path cost from start to current node
            tentative_g = G.nodes[node]["g_score"] + weight
            
            if tentative_g < G.nodes[neighbor]["g_score"]:
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["g_score"] = tentative_g
                
                #calculate h(n)
                h_n = manhattan_heuristic(neighbor, dest, G.nodes[neighbor]["maxspeed"])
                #calculate f(n)= g(n) + h(n)
                G.nodes[neighbor]["f_score"] = tentative_g + h_n
                heapq.heappush(pq, (G.nodes[neighbor]["f_score"], neighbor))
        
        step += 1
    return step





place_name = "Turin, Piedmont, Italy"
G = ox.graph_from_place(place_name, network_type="drive")

# Pre-processing: calcolo del peso (tempo di percorrenza) come nel file Dijkstra
for u, v, k, data in G.edges(data=True, keys=True):
    # Default speed se manca il dato
    maxspeed = 40 
    if "maxspeed" in data:
        ms = data["maxspeed"]
        if isinstance(ms, list):
            maxspeed = min([int(s) if s.isdigit() else 40 for s in ms])
        elif isinstance(ms, str) and ms.isdigit():
            maxspeed = int(ms)
    
    data["maxspeed"] = maxspeed
    # Peso = Lunghezza / Velocità (Tempo)
    data["weight"] = data["length"] / maxspeed


print("Running A* with Manhattan heuristic...")
results = []
for i in range(10):
    start = random.choice(list(G.nodes))
    dest = random.choice(list(G.nodes))    
    print(f"Start: {start}, Destination: {dest}")
    steps = astar(start, dest)
    results.append(steps)
    print(f"Run number {i+1} takes {steps} steps to convergence\n")
average_steps = sum(results) / len(results)
print(f"Average steps to convergence over 10 runs: {average_steps}")

