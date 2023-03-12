## CS7IS2 - Aritificial Intelligence - Assignment 1

#### Files:
1. Algorithms.py - Contains all the algorithms to solve the maze.
2. DataStructures.py - Contains the data sstructures such as stack and queue that is utilized in Algorithms.py
3. Maze.py - Contains class to Maze which initializes and defines the maze for our program.
4. main.py - The main program


#### Libraries used:
1. Pyamaze `(pip install pyamaze)`
2. Matplotlib `(pip install matplotlib)`

##### Command to run the code:
`python main.py -sx [sizex] -sy [sizey] -tx [targetx] -ty [targety] -del [delta] -loadmaze [mazename] -solver [solvername] -savemaze [mazename]`

##### Parameters:
1. sizex, sizey - Size of the maze in integers
2. targetx, targety - Location of the target/goal node in the maze
3. del - Delay provided to the trace path algorithm
4. loadmaze - Name of the maze to load
5. savemaze - Name of the maze you want to save
6. solver - Name of the solver to be used

##### Type of solvers (Put the exact name in the -solver parameter):
1. All
2. BFS
3. DFS
4. AStar
5. MDPVI
6. MDPPI
