import heapq
import math
import random
import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import osmnx as ox


def normalize_maxspeed(value) -> int:
    default_speed = 40
    if value is None:
        return default_speed
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if item == "walk":
                cleaned.append(1)
            else:
                text = str(item).strip()
                if text.endswith(" mph"):
                    text = text.replace(" mph", "")
                cleaned.append(int(text))
        return min(cleaned) if cleaned else default_speed
    if isinstance(value, str):
        if value == "walk":
            return 1
        text = value.strip()
        if text.endswith(" mph"):
            text = text.replace(" mph", "")
        return int(text)
    return int(value)


def prepare_graph(place_name: str):
    graph = ox.graph_from_place(place_name, network_type="drive")
    for edge in graph.edges:
        maxspeed_raw = graph.edges[edge].get("maxspeed")
        maxspeed = normalize_maxspeed(maxspeed_raw)
        graph.edges[edge]["maxspeed"] = maxspeed
        graph.edges[edge]["weight"] = graph.edges[edge]["length"] / maxspeed
        graph.edges[edge]["dijkstra_uses"] = 0
    return graph


def style_unvisited_edge(graph, edge):
    graph.edges[edge]["color"] = "gray"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 0.2


def style_visited_edge(graph, edge):
    graph.edges[edge]["color"] = "green"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 1


def style_active_edge(graph, edge):
    graph.edges[edge]["color"] = "red"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 1


def style_path_edge(graph, edge):
    graph.edges[edge]["color"] = "white"
    graph.edges[edge]["alpha"] = 1
    graph.edges[edge]["linewidth"] = 5


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


def dijkstra(graph, orig: int, dest: int) -> Tuple[Optional[int], Optional[List[int]]]:
    for node in graph.nodes:
        graph.nodes[node]["visited"] = False
        graph.nodes[node]["distance"] = float("inf")
        graph.nodes[node]["previous"] = None
        graph.nodes[node]["size"] = 0

    for edge in graph.edges:
        style_unvisited_edge(graph, edge)

    graph.nodes[orig]["distance"] = 0
    graph.nodes[orig]["size"] = 50
    graph.nodes[dest]["size"] = 50

    pq = [(0, orig)]
    step = 0

    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            return step, reconstruct_path(graph, orig, dest, algorithm="dijkstra")

        if graph.nodes[node]["visited"]:
            continue

        graph.nodes[node]["visited"] = True
        for edge in graph.out_edges(node):
            edge_key = (edge[0], edge[1], 0)
            style_visited_edge(graph, edge_key)
            neighbor = edge[1]
            weight = graph.edges[edge_key]["weight"]

            if graph.nodes[neighbor]["distance"] > graph.nodes[node]["distance"] + weight:
                graph.nodes[neighbor]["distance"] = graph.nodes[node]["distance"] + weight
                graph.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (graph.nodes[neighbor]["distance"], neighbor))

                for edge2 in graph.out_edges(neighbor):
                    style_active_edge(graph, (edge2[0], edge2[1], 0))

        step += 1

    return None, None


def reconstruct_path(graph, orig: int, dest: int, algorithm: Optional[str] = None) -> Optional[List[int]]:
    for node in graph.nodes:
        graph.nodes[node]["size"] = 0

    for edge in graph.edges:
        style_unvisited_edge(graph, edge)

    route = [dest]
    curr = dest
    while curr != orig:
        prev = graph.nodes[curr]["previous"]
        if prev is None:
            return None
        style_path_edge(graph, (prev, curr, 0))
        if algorithm:
            key = f"{algorithm}_uses"
            graph.edges[(prev, curr, 0)][key] = graph.edges[(prev, curr, 0)].get(key, 0) + 1
        curr = prev
        route.append(curr)

    route.reverse()
    return route


def generate_pairs(graph, num_trials: int, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    nodes = list(graph.nodes)
    pairs = []
    for _ in range(num_trials):
        start = random.choice(nodes)
        end = random.choice(nodes)
        while end == start:
            end = random.choice(nodes)
        pairs.append((start, end))
    return pairs


def build_dijkstra_collage(
    graph,
    pairs,
    show: bool = True,
    save_path: Optional[str] = None,
    title_prefix: str = "Trial",
):
    cols = 5
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, max(5, rows * 5)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (start_i, end_i) in enumerate(pairs):
        dijkstra(graph, start_i, end_i)
        plot_graph(graph, ax=axes[idx], show=False, close=False)
        axes[idx].set_title(f"{title_prefix} {idx + 1}", color="white", fontsize=10)

    for idx in range(len(pairs), len(axes)):
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


def run_dijkstra_demo(place_name: str = "Aosta, Aosta, Italy", num_trials: int = 10, seed: Optional[int] = 123):
    graph = prepare_graph(place_name)
    pairs = generate_pairs(graph, num_trials=num_trials, seed=seed)
    iterations = []

    print("Running Dijkstra")
    print("City:", place_name)
    print("Nodes:", len(graph.nodes))
    print("Edges:", len(graph.edges))
    print("Trials:", num_trials)

    for i, (start, end) in enumerate(pairs, start=1):
        it, _ = dijkstra(graph, start, end)
        iterations.append(it)
        print(f"Trial {i:2d} | Iterations={it}")

    valid_iterations = [value for value in iterations if value is not None]
    if valid_iterations:
        print("\nAverage iterations:", sum(valid_iterations) / len(valid_iterations))

    build_dijkstra_collage(graph, pairs, show=True, save_path=None)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dijkstra demo with collage visualization")
    parser.add_argument("--city", type=str, default="Turin, Piedmont, Italy", help="City/place name for OSM graph")
    parser.add_argument("--trials", type=int, default=10, help="Number of random start/end trials")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible pairs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dijkstra_demo(place_name=args.city, num_trials=args.trials, seed=args.seed)

