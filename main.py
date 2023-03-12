import matplotlib.pyplot as plt
from Maze import *
import math
import argparse

def ManhattanHeuristic(node1, node2):
    return (abs(node1[0] - node2[0]) + abs(node2[1] - node2[1]))

def SquaredDistance(node1, node2):
    return math.sqrt(math.pow((node1[0] - node2[0]),2.0) + math.pow(abs(node1[1] - node2[1]), 2.0))    

ProgramMetrics = dict()
ProgramMetrics["Maze"] = dict()
ProgramMetrics["Maze"]["SizeX"] = 0
ProgramMetrics["Maze"]["SizeY"] = 0
ProgramMetrics["Maze"]["TargetX"] = None
ProgramMetrics["Maze"]["TargetY"] = None
ProgramMetrics["Maze"]["SaveMaze"] = None
ProgramMetrics["Maze"]["LoadMaze"] = None
ProgramMetrics["Maze"]["Solver"] = "All"

if __name__ == "__main__": 
    parser = argparse.ArgumentParser( prog='MazeSolver',
                        description='Solves mazes using different algorithm',
                        epilog='For assignment-1 - AI')
    parser.add_argument('-g', '--generate')
    parser.add_argument('-sx', '--sizex')
    parser.add_argument('-sy', '--sizey')
    parser.add_argument('-tx', '--targetx')
    parser.add_argument('-ty', '--targety')
    parser.add_argument('-snx', '--startnodex')
    parser.add_argument('-sny', '--startnodey')
    parser.add_argument('-del', '--delay')
    parser.add_argument('-savemaze', '--savemaze')
    parser.add_argument('-loadmaze', '--loadmaze')
    parser.add_argument('-solver', '--solver')
    args = parser.parse_args()
    if args.sizex != None and args.sizey != None:
        ProgramMetrics["Maze"]["SizeX"] = int(args.sizex)
        ProgramMetrics["Maze"]["SizeY"] = int(args.sizey)
    else:
        print("No size metrics were provided in command line hence the assumed maze-size is (30x30)")
        ProgramMetrics["Maze"]["SizeX"] = 30
        ProgramMetrics["Maze"]["SizeY"] = 30
    if args.targetx != None and args.targety != None:
        ProgramMetrics["Maze"]["TargetX"] = int(args.targetx)
        ProgramMetrics["Maze"]["TargetY"] = int(args.targety)
    
    if args.delay != None:
        ProgramMetrics["Maze"]["Delay"] = int(args.delay)
    else:
        print("No delay metric was provided in the command line hence the assumed delay is 100")
        ProgramMetrics["Maze"]["Delay"] = 100
    
    if args.savemaze != None:
        ProgramMetrics["Maze"]["SaveMaze"] = True
        ProgramMetrics["Maze"]["SaveMazeName"] = args.savemaze
    if args.loadmaze != None:
        ProgramMetrics["Maze"]["LoadMaze"] = True
        ProgramMetrics["Maze"]["LoadMazeName"] = args.loadmaze
    if args.solver != None:
        ProgramMetrics["Maze"]["Solver"] = args.solver

    BFSSolver = BFS()
    BFSSolver.SetSolverColor(COLOR.red)
    BFSSolver.SetSolverFilled(True)
    BFSSolver.SetName("BFS")
    DFSSolver = DFS()
    DFSSolver.SetSolverColor(COLOR.cyan)
    DFSSolver.SetSolverFilled(True)
    DFSSolver.SetName("DFS")
    AStarSolver = AStar()
    AStarSolver.SetHeuristicFunction(SquaredDistance)
    AStarSolver.SetSolverColor(COLOR.dark)
    AStarSolver.SetSolverFilled(False)
    AStarSolver.SetName("AStar")
    AStarSolverManhattan = AStar()
    AStarSolverManhattan.SetHeuristicFunction(ManhattanHeuristic)
    AStarSolverManhattan.SetSolverColor(COLOR.green)
    AStarSolverManhattan.SetSolverFilled(False)
    AStarSolverManhattan.SetName("AStar")
    MDPVIInstance = MDPVI()
    MDPVIInstance.SetMaximumError(0.001)
    MDPVIInstance.SetDiscount(0.9)
    MDPVIInstance.SetReward(-10)
    MDPVIInstance.SetSolverShape("arrow")
    MDPVIInstance.SetSolverColor(COLOR.yellow)
    # MDPVIInstance.SetSolverFilled(False)
    MDPVIInstance.Strategy = MDPSolverStrategy.ValueIteration
    MDPPIInstance = MDPPI()
    MDPPIInstance.SetMaximumError(0.001)
    MDPPIInstance.SetDiscount(0.9)
    MDPPIInstance.SetReward(1000)
    MDPPIInstance.SetSolverShape("arrow")
    MDPPIInstance.SetSolverColor(COLOR.blue)
    # MDPPIInstance.SetSolverFilled(True)
    MDPPIInstance.Strategy = MDPSolverStrategy.PolicyIteration
    FinalMaze = Maze()
    if ProgramMetrics["Maze"]["SaveMaze"] != None:
        FinalMaze.SaveMaze(True, ProgramMetrics["Maze"]["SaveMazeName"])
    if ProgramMetrics["Maze"]["LoadMaze"] != None:
        FinalMaze.LoadMaze(True, ProgramMetrics["Maze"]["LoadMazeName"])
    if ProgramMetrics["Maze"]["TargetX"] != None and ProgramMetrics["Maze"]["TargetY"] != None:
        FinalMaze.SetGoalNode((ProgramMetrics["Maze"]["TargetX"], ProgramMetrics["Maze"]["TargetY"]))
    FinalMaze.SetAgentDelay(ProgramMetrics["Maze"]["Delay"])
    FinalMaze.CreateMaze(size=(ProgramMetrics["Maze"]["SizeX"] ,ProgramMetrics["Maze"]["SizeY"] ))
    SolverList = ["BFS", "DFS", "AStar", "MDPVI", "MDPPI"]
    FinalList = []

    if ProgramMetrics["Maze"]["Solver"] == "All":
        for solver_name in SolverList:
            FinalList.append(solver_name)
    else:
        solver_list_args = ProgramMetrics["Maze"]["Solver"].split(",")
        print(" Args " ,solver_list_args)
        for solver in solver_list_args:
            FinalList.append(solver)
    print( "Final List" , FinalList)
    if "BFS" in FinalList:
        print("Adding BFS")
        FinalMaze.AddSolver(BFSSolver)
    if "DFS" in FinalList:
        FinalMaze.AddSolver(DFSSolver)
    if "AStar" in FinalList:
        FinalMaze.AddSolver(AStarSolver)
        FinalMaze.AddSolver(AStarSolverManhattan)
    if "MDPVI" in FinalList:
        FinalMaze.AddSolver(MDPVIInstance)
    if "MDPPI" in FinalList:
        FinalMaze.AddSolver(MDPPIInstance)
    try:
        FinalMaze.SolveMaze()
    except KeyError:
        print("Key Error")
        # FinalMaze.PlotSolvedMaze()
    if "BFS" in FinalList:
        FinalMaze.AddLabel(f"BFS TD: ",      round(BFSSolver.GetTimeDifference(), 3))
        FinalMaze.AddLabel(f"BFS PL: ",       BFSSolver.GetFinalPathLength()) 
        FinalMaze.AddLabel(f"BFS SPL: ",       BFSSolver.GetSearchPathLength()) 
    if "DFS" in FinalList:
        FinalMaze.AddLabel(f"DFS TD: ",      round(DFSSolver.GetTimeDifference(), 3))
        FinalMaze.AddLabel(f"DFS PL: ",       DFSSolver.GetFinalPathLength())
        FinalMaze.AddLabel(f"DFS SPL: ",       DFSSolver.GetSearchPathLength()) 
    if "AStar" in FinalList:
        FinalMaze.AddLabel(f"AStar TD: ",    round(AStarSolver.GetTimeDifference(), 3))
        FinalMaze.AddLabel(f"AStar MHTN TD: ",    round(AStarSolverManhattan.GetTimeDifference(), 3))
        FinalMaze.AddLabel(f"AStar PL: ",     AStarSolver.GetFinalPathLength()) 
        FinalMaze.AddLabel(f"AStar SPL: ",       AStarSolver.GetSearchPathLength()) 
        FinalMaze.AddLabel(f"AStar MHTN PL: ",     AStarSolverManhattan.GetFinalPathLength()) 
        FinalMaze.AddLabel(f"AStar MNTH SPL: ",       AStarSolverManhattan.GetSearchPathLength()) 
    if "MDPVI" in FinalList:
        FinalMaze.AddLabel(f"MDP-VI TD: ",   round(MDPVIInstance.GetTimeDifference(), 3))
        FinalMaze.AddLabel(f"MDP-VI PL: ",    MDPVIInstance.GetFinalPathLength()) 
    if "MDPPI" in FinalList:
        FinalMaze.AddLabel(f"MDP-PI TD: ",   round(MDPPIInstance.GetTimeDifference(), 3))
        FinalMaze.AddLabel(f"MDP-PI PL: ",    MDPPIInstance.GetFinalPathLength()) 

    FinalMaze.Run()
    
    """
        After all this mess, finally plot some graphs..
    """
    # Plot the time bar graph
    if ProgramMetrics["Maze"]["Solver"] == "All":
        Algorithms = ["BFS", "DFS", "A-Star", "MDP(VI)", "MDP(PI)"]
        Values = [
            round(BFSSolver.GetTimeDifference(), 3),
            round(DFSSolver.GetTimeDifference(), 3),
            round(AStarSolver.GetTimeDifference(), 3),
            round(MDPVIInstance.GetTimeDifference(), 3),
            round(MDPPIInstance.GetTimeDifference(), 3)
        ]
        fig = plt.figure(dpi=150)
        if ProgramMetrics["Maze"]["LoadMaze"] == True:
            plt.title(f"Time benchmark for maze '{ProgramMetrics['Maze']['LoadMazeName']}'")
        else:
            plt.title(f"Time benchmark for maze of size ({ProgramMetrics['Maze']['SizeX']}, {ProgramMetrics['Maze']['SizeY']}) and target ({ProgramMetrics['Maze']['TargetX']},{ProgramMetrics['Maze']['TargetY']})")
        plt.ylabel("Time (t)")
        plt.xlabel("Algorithms")
        plt.bar(Algorithms, Values)
        plt.show()
        plt.close(fig)

        Values = [
            BFSSolver.GetFinalPathLength(),
            DFSSolver.GetFinalPathLength(), 
            AStarSolver.GetFinalPathLength(),
            MDPVIInstance.GetFinalPathLength(), 
            MDPPIInstance.GetFinalPathLength()
        ]
        fig = plt.figure(dpi=150)
        if ProgramMetrics["Maze"]["LoadMaze"] == True:
            plt.title(f"Path benchmark for maze '{ProgramMetrics['Maze']['LoadMazeName']}'")
        else:
            plt.title(f"Path benchmark for maze of size ({ProgramMetrics['Maze']['SizeX']}, {ProgramMetrics['Maze']['SizeY']}) and target ({ProgramMetrics['Maze']['TargetX']},{ProgramMetrics['Maze']['TargetY']})")
        plt.ylabel("Path Length (PL)")
        plt.xlabel("Algorithms")
        plt.bar(Algorithms, Values)
        plt.show()
        plt.close(fig)