#!/bin/bash
#this bash script runs first Dijkstra and then A* in the project virtualenv

echo "Starting batch execution: Dijkstra then A* on the same cities"

#activate the virtual environment
source ../.venv/bin/activate

#execute both algorithms and capture their outputs for later comparison

echo "Executing Dijkstra..."
python Dijkstra.py > dijkstra_output.log 2>&1

echo "Executing A*..."
python Astar.py > astar_output.log 2>&1

echo ""
echo "--- SIMULATION COMPLETED ---"
deactivate
