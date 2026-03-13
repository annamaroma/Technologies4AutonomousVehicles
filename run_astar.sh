#!/bin/bash
#this bash script is used to run easily the algorithms without having to activate the virtual environment and run the python script manually every time

#comunicate the user that the script is starting
echo "Starting A* algorithm execution... :) "

#activate the virtual environment
source venv/bin/activate

#run Astar algorithm
python assignment1/Astar_rifatto.py

#check for errors
if [ $? -ne 0 ]; then
    echo "Error encountered while running the A* algorithm. :\ "
    echo "Please check the error messages above and try again."
    deactivate
else
    # PER ORA: deactivate
    deactivate
    echo "A* algorithm execution completed... :) "
fi