# Technologies4AutonomousVehicles

Repository containing assignments for the Technologies for Autonomous Vehicles course, focusing on **path planning algorithms** and **lane detection techniques**.

---

## 📋 Assignments Overview

### Assignment 1: Global Path Planning (Dijkstra vs A*)
Compares Dijkstra and A* algorithms with multiple heuristics (Manhattan, Euclidean, Haversine) for finding optimal routes on urban road networks of Turin and Aosta.

### Assignment 2: Lane Detection (GOLD Algorithm)
Implements the GOLD (Gradient-based Lane Detection) algorithm to detect road lanes from vehicle camera images, including bird's eye view transformation and camera calibration.

---

## 📂 Project Structure

```
Technologies4AutonomousVehicles/
├── README.md
├── assignment1/
│   ├── Dijkstra.py          # Dijkstra shortest path implementation
│   ├── Astar.py             # A* search with multiple heuristics
│   ├── run_simulation.sh     # Automation script for path planning
│   ├── cache/               # Cached graph data
│   ├── results/             # Output visualizations and logs
│   └── .venv/               # Python virtual environment
├── assignment2/
│   ├── main.m               # MATLAB main script for lane detection
│   ├── Latex_report/        # LaTeX project report
│   ├── results/             # Output images and analysis
│   └── *.pdf                # Assignment documents and papers
└── ...
```

---

## ⚠️ Prerequisites

### Assignment 1 (Python)

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

### Assignment 2 (MATLAB)

Requires MATLAB with Computer Vision Toolbox. Ensure the image dataset is available in the expected path (`archive/044/camera/front_camera/`).

---

## 🚀 How to Run

### Assignment 1: Path Planning

```bash
cd assignment1
./run_simulation.sh
```

**What the script does:**
1. Activates the virtual environment.
2. Runs Dijkstra and A* on the chosen node pairs.
3. Stops early on errors to avoid inconsistent results.
4. Generates logs for both successful runs and errors.

**Expected Outputs:**
* **Terminal**: average cost, distance, iterations, and optimality check.
* **`assignment1/results/`**: trajectory visualizations and result files.

---

### Assignment 2: Lane Detection

```matlab
% In MATLAB, navigate to assignment2 directory and run:
main
```

**What the script does:**
1. Loads camera images from the dataset.
2. Performs bird's eye view transformation using camera calibration.
3. Applies GOLD lane detection algorithm.
4. Generates detected lane visualizations.

**Expected Outputs:**
* **`assignment2/results/`**: processed images with detected lanes and analysis.

---
