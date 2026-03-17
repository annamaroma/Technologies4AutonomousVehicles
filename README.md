# Technologies4AutonomousVehicles

## Global Path Planning: Dijkstra vs A*

This repository compares Dijkstra and A* (Manhattan, Euclidean, Haversine heuristics) on urban road networks of Turin and Aosta.

## 📂 Project Structure
* **`assignment1/Dijkstra.py`**: Computes the optimal travel cost/time (ground truth).
* **`assignment1/Astar.py`**: Executes A* search on the same road graph using multiple heuristics.
* **`assignment1/run_simulation.sh`**: Automates environment activation, algorithm execution, and validation.

---
## ⚠️ Prerequisites

From the repository root, run:

```bash
cd assignment1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then make the runner executable:

```bash
chmod +x run_simulation.sh
```

---
## 🚀 How to Run

```bash
cd assignment1
./run_simulation.sh
```

---
## ⚙️ What the script does
1. Activates the virtual environment.
2. Runs Dijkstra and A* on the chosen node pairs.
3. Stops early on errors to avoid inconsistent results.
4. Output files contain the log in case either of success and errors encourred during simulation.

---
## 📊 Expected Outputs
* **Terminal**: average cost, distance, iterations, and optimality check.
* **`assignment1/results/`**: generated trajectory visualizations and result files.

---
