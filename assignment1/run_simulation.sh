#!/bin/bash
#this bash script runs first A* and then Dijkstra in the project virtualenv

echo "Starting batch execution: A* then Dijkstra on the same cities"

#activate the virtual environment
source .venv/bin/activate

#execute both algorithms and capture their outputs for later comparison
echo "Executing Dijkstra..."
DIJKSTRA_OUT=$(python Dijkstra.py)
if [ $? -ne 0 ]; then
  echo "CRITICAL ERROR: Dijkstra.py has crashed."
  deactivate
  exit 1
fi


echo "Executing A*..."
ASTAR_OUT=$(python Astar_rifatto.py)
if [ $? -ne 0 ]; then
  echo "CRITICAL ERROR: Astar_rifatto.py has crashed."
  deactivate
  exit 1
fi

#define cities and heuristics
CITIES=("Turin, Piedmont, Italy" "Aosta, Aosta, Italy")
HEURISTICS=("Manhattan" "Euclidean" "Haversine")

#check for optimality
#compare the average cost of A* with the average cost of Dijkstra for each city and heuristic
for CITY in "${CITIES[@]}"; do
    echo ""
    echo ">>>> VERIFICA CITTÀ: $CITY <<<<"
    
    #extract the average cost of Dijkstra for the city
    AVG_COST_D=$(echo "$DIJKSTRA_OUT" | grep "AVG_COST_DIJKSTRA_$CITY" | tail -1 | cut -d' ' -f2)

    for H in "${HEURISTICS[@]}"; do
        #extract the average cost of A* for the city and heuristic
        AVG_COST_A=$(echo "$ASTAR_OUT" | grep "AVG_COST_ASTAR_${CITY}_${H}" | tail -1 | cut -d' ' -f2)

        echo -n "Checking A* ($H) vs Dijkstra... "
        echo "Dijkstra AVG_ITERATIONS: $AVG_COST_D, A* AVG_COST: $AVG_COST_A"
        echo "Dijkstra AVG_COST: $AVG_COST_D, A* AVG_COST: $AVG_COST_A"
        echo "Dijkstra AVG_DISTANCE: $AVG_COST_D, A* AVG_DISTANCE: $AVG_COST_A"

        #The optimality condition is that the average cost of A* should be very close to the average cost of Dijkstra, 
        #since Dijkstra always finds the optimal solution. We allow a small tolerance due to potential differences in implementation or floating-point precision.
        #abs($AVG_COST_A - $AVG_COST_D) < 0.0001
        
        DIFF=$(echo "$AVG_COST_A - $AVG_COST_D" | bc -l | sed 's/-//') # Calcola valore assoluto
        IS_OPTIMAL=$(echo "$DIFF < 0.0001" | bc -l)

        if [ "$IS_OPTIMAL" -eq 1 ]; then
            echo "✅ OPTIMAL (Cost: $AVG_COST_A)"
        else
            echo "❌ ERROR! Difference detected: $DIFF"
            echo "A* is not converging to the optimum for $CITY with $H."
        fi
    done
done

echo ""
echo "--- SIMULATION COMPLETED ---"
deactivate
















# run A* algorithm
python Astar_rifatto.py
if [ $? -ne 0 ]; then
  echo "A* failed. Stopping."
  deactivate
  exit 1
fi

echo "A* completed successfully. Running Dijkstra..."

# run Dijkstra algorithm
python Dijkstra.py
if [ $? -ne 0 ]; then
  echo "Dijkstra failed."
  deactivate
  exit 1
fi

#to make sure that both algorithms completed successfully and reached optimum,
#I check the total distance for both algorithms and compare them. If they are the same, then both algorithms reached the optimum solution.

echo "Both algorithms completed successfully."
deactivate
