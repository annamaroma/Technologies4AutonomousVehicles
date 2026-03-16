#!/bin/bash
#this bash script runs first Dijkstra and then A* in the project virtualenv

echo "Starting batch execution: Dijkstra then A* on the same cities"

#activate the virtual environment
source ../.venv/bin/activate

#execute both algorithms and capture their outputs for later comparison
echo "Executing Dijkstra..."
#DIJKSTRA_OUT=$(python Dijkstra.py)
#if [ $? -ne 0 ]; then
#  echo "CRITICAL ERROR: Dijkstra.py has crashed."
#  deactivate
#  exit 1
#fi
python Dijkstra.py > dijkstra_output.log 2>&1


echo "Executing A*..."
#ASTAR_OUT=$(python Astar.py)
#if [ $? -ne 0 ]; then
#  echo "CRITICAL ERROR: Astar.py has crashed."
#  deactivate
#  exit 1
#fi
python Astar.py > astar_output.log 2>&1

#scrtivo i risultati in un file di log
LOG_FILE="./simulation_results.log"
echo "Simulation Results - Dijkstra and A* Comparison" > "$LOG_FILE"

#for CITY in "${CITIES[@]}"; do
#    echo ""
#    echo ">>>> Log city: $CITY <<<<" >> "$LOG_FILE"
    
    #extract the average cost of Dijkstra for the city
#    AVG_COST_D=$(echo "$DIJKSTRA_OUT" | grep "AVG_COST_DIJKSTRA_$CITY" | tail -1 | cut -d' ' -f2)
    #carico avg cost di Dijkstra nel file di log
#    echo "City: $CITY | Dijkstra Average Cost: $AVG_COST_D" >> "$LOG_FILE"

#    for H in "${HEURISTICS[@]}"; do
        #extract the average cost of A* for the city and heuristic
#        AVG_COST_A=$(echo "$ASTAR_OUT" | grep "AVG_COST_ASTAR_${CITY}_${H}" | tail -1 | cut -d' ' -f2)
        #carico avg cost di A* nel file di log
#        echo "City: $CITY | Heuristic: $H | A* Average Cost: $AVG_COST_A" >> "$LOG_FILE"
#    done
#done

echo ""
echo "--- SIMULATION COMPLETED ---"
deactivate
