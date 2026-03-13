import random
import argparse
from typing import Dict, List, Optional, Tuple
import os
import matplotlib.pyplot as plt
import osmnx as ox

from Astar import astar, plot_graph as plot_graph_astar
from Dijkstra import dijkstra, plot_graph as plot_graph_dijkstra

# Compute total cost (time) to verify admissibility of A* paths versus Dijkstra.
def calculate_path_cost(graph, path: Optional[List[int]]) -> float:
    """Calculate the sum of the 'weight' (time) of all edges in a path."""
    if not path or len(path) < 2:
        return float('inf')
    
    total_cost = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        # Get the edge weight between u and v
        total_cost += graph.edges[(u, v, 0)]["weight"]
        
    return total_cost

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

# Ensure heuristic can use the maximum speed present in the graph
def prepare_graph(place_name: str):
    graph = ox.graph_from_place(place_name, network_type="drive")
    max_speeds_found =[]

    for edge in graph.edges:
        maxspeed_raw = graph.edges[edge].get("maxspeed")
        maxspeed = normalize_maxspeed(maxspeed_raw)
        graph.edges[edge]["maxspeed"] = maxspeed
        graph.edges[edge]["weight"] = graph.edges[edge]["length"] / maxspeed
        max_speeds_found.append(maxspeed)
        
    # Save the global maximum speed as a graph attribute (dynamic!)
    graph.graph["max_speed"] = max(max_speeds_found) if max_speeds_found else 40
    return graph

def generate_pairs(graph, num_trials: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
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


def avg_valid(values: List[Optional[int]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def run_experiment(
    place_name: str = "Aosta, Aosta, Italy",
    num_trials: int = 10,
    seed: Optional[int] = None,
    heuristic_mode: str = "all",
):
    graph = prepare_graph(place_name)
    pairs = generate_pairs(graph, num_trials=num_trials, seed=seed)

    all_heuristics = ["manhattan", "euclidean", "haversine"]
    heuristics = all_heuristics if heuristic_mode == "all" else [heuristic_mode]
    rows: List[Dict[str, Optional[int]]] = []

    print("Comparing Dijkstra vs A* on same random pairs")
    print("Max speed:", graph.graph["max_speed"])
    print("City:", place_name)
    print("Nodes:", len(graph.nodes))
    print("Edges:", len(graph.edges))
    print("Trials:", num_trials)

    print("A* Heuristic mode:", heuristic_mode)

    for idx, (start, end) in enumerate(pairs, start=1):
            d_it, d_path = dijkstra(graph, start, end)
            d_cost = calculate_path_cost(graph, d_path)

            row: Dict[str, Optional[float]] = {
                "trial": idx,
                "start": start,
                "end": end,
                "dijkstra_it": d_it,
                "dijkstra_cost": d_cost,
            }

            print(f"\n--- Trial {idx:2d} | start={start} end={end} ---")
            print(f"Dijkstra   -> Iter: {d_it:5d} | Cost (Time): {d_cost:.4f}")

            for h in heuristics:
                a_it, a_path = astar(graph, start, end, heuristic_kind=h)
                a_cost = calculate_path_cost(graph, a_path)
                
                row[f"astar_{h}_it"] = a_it
                row[f"astar_{h}_cost"] = a_cost
                
                # Check optimality of the path found by A* against Dijkstra
                if a_cost > d_cost:
                    status = "Sub-optimal (Admissibility violated: path cost higher than optimal)"
                elif abs(a_cost - d_cost) <= 0:
                    status = "Optimal (Same cost as Dijkstra)"
                else:
                    status = "Anomalous (A* found a path better than Dijkstra)"

                print(f"A* {h.capitalize():<9} -> Iter: {a_it:5d} | Costo (Tempo): {a_cost:.4f} {status}")
                
            rows.append(row)

    # compute averages
    d_it_avg = avg_valid([r["dijkstra_it"] for r in rows])
    d_cost_avg = avg_valid([r["dijkstra_cost"] for r in rows])
    
    print("\n" + "="*40)
    print("avg iterations:")
    print(f"Dijkstra   : {d_it_avg:.1f}")
    
    summary = {
        "max_speed_city": graph.graph["max_speed"],
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "avg_iterations_dijkstra": d_it_avg,
        "avg_cost_dijkstra": d_cost_avg
    }

    if heuristic_mode == "all":
        man_it_avg = avg_valid([r["astar_manhattan_it"] for r in rows])
        euc_it_avg = avg_valid([r["astar_euclidean_it"] for r in rows])
        hav_it_avg = avg_valid([r["astar_haversine_it"] for r in rows])
        
        man_cost_avg = avg_valid([r["astar_manhattan_cost"] for r in rows])
        euc_cost_avg = avg_valid([r["astar_euclidean_cost"] for r in rows])
        hav_cost_avg = avg_valid([r["astar_haversine_cost"] for r in rows])

        print(f"A* Manhattan : {man_it_avg:.1f}")
        print(f"A* Euclidean : {euc_it_avg:.1f}")
        print(f"A* Haversine : {hav_it_avg:.1f}")

        print("\naverage cost (time):")
        print(f"Dijkstra   : {d_cost_avg:.4f}")
        print(f"A* Manhattan : {man_cost_avg:.4f}")
        print(f"A* Euclidean : {euc_cost_avg:.4f}")
        print(f"A* Haversine : {hav_cost_avg:.4f}")

        summary.update({
            "avg_iterations_astar_manhattan": man_it_avg,
            "avg_iterations_astar_euclidean": euc_it_avg,
            "avg_iterations_astar_haversine": hav_it_avg,
            "avg_cost_astar_manhattan": man_cost_avg,
            "avg_cost_astar_euclidean": euc_cost_avg,
            "avg_cost_astar_haversine": hav_cost_avg
        })

    else:
        sel_it_avg = avg_valid([r[f"astar_{heuristic_mode}_it"] for r in rows])
        sel_cost_avg = avg_valid([r[f"astar_{heuristic_mode}_cost"] for r in rows])
        
        print(f"A* {heuristic_mode.capitalize()} : {sel_it_avg:.1f}")
        
        print("\naverage cost (time):")
        print(f"Dijkstra   : {d_cost_avg:.4f}")
        print(f"A* {heuristic_mode.capitalize()} : {sel_cost_avg:.4f}")

        summary.update({
            f"avg_iterations_astar_{heuristic_mode}": sel_it_avg,
            f"avg_cost_astar_{heuristic_mode}": sel_cost_avg
        })
    print("="*40 + "\n")
    
    import csv
    csv_fields = list(rows[0].keys()) if rows else []
    def city_to_filename(city):
        return city.lower().replace(",", "").replace(" ", "_")

    city_part = city_to_filename(place_name)

    results_dir = os.path.join("results", city_part)
    pics_dir = os.path.join(results_dir, "pics")
    os.makedirs(pics_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"results_{city_part}.csv")
    summary_file = os.path.join(results_dir, f"summary_{city_part}.json")

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        rounded_rows = []
        int_columns = {
            "trial",
            "start",
            "end",
            "dijkstra_it",
            "astar_manhattan_it",
            "astar_euclidean_it",
            "astar_haversine_it",
        }
        for row in rows:
            rounded_row = {}
            for k, v in row.items():
                if k in int_columns and v is not None:
                    rounded_row[k] = int(v)
                elif isinstance(v, (int, float)):
                    try:
                        rounded_row[k] = round(float(v), 2)
                    except Exception:
                        rounded_row[k] = v
                else:
                    rounded_row[k] = v
            rounded_rows.append(rounded_row)
        writer.writerows(rounded_rows)
    print(f"Trial results saved to {results_file}")

    import json
    rounded_summary = {k: round(float(v), 2) if isinstance(v, (int, float)) else v for k, v in summary.items()}
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(rounded_summary, f, indent=2)
    print(f"Summary averages saved to {summary_file}")


    dijkstra_file = os.path.join(pics_dir, f"collage_dijkstra_{city_part}.png")
    save_collage_dijkstra(graph, pairs, save_path=dijkstra_file)
    print(f"\nCollage saved: {dijkstra_file}")

    if heuristic_mode == "all":
        for h in all_heuristics:
            astar_file = os.path.join(pics_dir, f"collage_astar_{h}_{city_part}.png")
            save_collage_astar(graph, pairs, save_path=astar_file, heuristic_kind=h)
            print(f"Collage saved: {astar_file}")
    else:
        astar_file = os.path.join(pics_dir, f"collage_astar_{heuristic_mode}_{city_part}.png")
        save_collage_astar(graph, pairs, save_path=astar_file, heuristic_kind=heuristic_mode)
        print(f"Collage saved: {astar_file}")

# save collage of paths found by Dijkstra
def save_collage_dijkstra(graph, pairs, save_path: str):
    cols = 5
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, max(5, rows * 5)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (start_i, end_i) in enumerate(pairs):
        dijkstra(graph, start_i, end_i)
        plot_graph_dijkstra(graph, ax=axes[idx], show=False, close=False)
        axes[idx].set_title(f"Trial {idx + 1}", color="white", fontsize=10)

    for idx in range(len(pairs), len(axes)):
        axes[idx].axis("off")
        axes[idx].set_facecolor("black")

    fig.patch.set_facecolor("black")
    plt.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=200, bbox_inches="tight")
    plt.close(fig)

# save collage of paths found by A*
def save_collage_astar(graph, pairs, save_path: str, heuristic_kind: str = "euclidean"):
    cols = 5
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, max(5, rows * 5)))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (start_i, end_i) in enumerate(pairs):
        astar(graph, start_i, end_i, heuristic_kind=heuristic_kind)
        plot_graph_astar(graph, ax=axes[idx], show=False, close=False)
        axes[idx].set_title(f"Trial {idx + 1}", color="white", fontsize=10)

    for idx in range(len(pairs), len(axes)):
        axes[idx].axis("off")
        axes[idx].set_facecolor("black")

    fig.patch.set_facecolor("black")
    plt.tight_layout()
    fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run stats and save path collages for Dijkstra and A*")
    parser.add_argument("--city", type=str, default="Aosta, Aosta, Italy", help="City/place name for OSM graph")
    # different city choice "Turin, Piedmont, Italy"
    parser.add_argument("--trials", type=int, default=10, help="Number of random start/end trials")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible pairs")
    parser.add_argument(
        "--heuristic",
        type=str,
        default="all",
        choices=["all", "manhattan", "euclidean", "haversine"],
        help="A* heuristic for stats/collage. 'all' keeps full comparison for stats",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        place_name=args.city,
        num_trials=args.trials,
        seed=args.seed,
        heuristic_mode=args.heuristic,
    )
