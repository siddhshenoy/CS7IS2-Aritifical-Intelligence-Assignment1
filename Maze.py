from pyamaze import maze, agent, COLOR, textLabel
from Algorithms import *

class Maze:
    def __init__(self):
        self.nodes = []
        self.maze = None
        self.agent = []
        self.agentDelay = 50
        self.SolverList = []
        self.pathList = []
        self.goal_node = None
        self.shouldSaveMaze = False
        self.saveMazeName = ""
        self.shouldLoadMaze = False
        self.loadMazeName = ""
    
    def LoadMaze(self, load_maze, load_maze_name):
        self.shouldLoadMaze = True
        self.loadMazeName = load_maze_name

    def SaveMaze(self, save_maze, save_maze_name):
        self.shouldSaveMaze = True
        self.saveMazeName = save_maze_name
        print("Save maze set")

    def SetGoalNode(self, goal_node):
        self.goal_node = goal_node

    def CreateMaze(self, size = (5, 5)):
        self.maze = maze(size[0], size[1])
        saveMaze = None
        loadMaze = None
        if self.shouldSaveMaze == True:
            saveMaze = self.saveMazeName
            print(f"Trying to save maze with name {self.saveMazeName}")
        if self.shouldLoadMaze == True:
            loadMaze = self.loadMazeName
        if self.shouldSaveMaze == True and self.shouldLoadMaze == True:
            print("The maze cannot load and save mazes at the same time..")
        else: 
            if self.goal_node != None:
                self.maze.CreateMaze(x = self.goal_node[0], y = self.goal_node[1],loopPercent=100, theme='dark', saveMaze=saveMaze,loadMaze=loadMaze)
            else:
                self.maze.CreateMaze(loopPercent=100, theme='dark', saveMaze=saveMaze,loadMaze=loadMaze)
    
    def SetAgentDelay(self, delay):
        self.agentDelay = delay

    def AddSolver(self, solver):
        self.SolverList.append(solver)

    def SetAgentDelay(self, agentDelay):
        self.agentDelay = agentDelay

    def SolveMaze(self):
        for i in range(0, len(self.SolverList)):
            self.SolverList[i].Solve(self.maze, self.maze._goal)
            path = self.SolverList[i].GetFinalPath()
            self.pathList.append(path)
            self.agent.append(agent(self.maze,
                            footprints=True,color=self.SolverList[i].GetSolverColor(),
                            shape=self.SolverList[i].GetSolverShape(),
                            filled=self.SolverList[i].GetSolverFilled()
                        ))  
            
        for i in range(0, len(self.agent)):
            self.maze.tracePath({self.agent[i]:self.pathList[i]},delay=self.agentDelay)
            #Label = textLabel(self.SolverList[i].GtName(), self.SolverList[i].Get)
    def PlotSolvedMaze(self):
        for i in range(0, len(self.SolverList)):
            path = self.SolverList[i].GetFinalPath()
            print(path)
            self.pathList.append(path)
            self.agent.append(agent(self.maze,
                            footprints=True,color=self.SolverList[i].GetSolverColor(),
                            shape='square',
                            filled=False)) 
    def AddLabel(self, label, label_value):
        l = textLabel(self.maze, label, label_value)
        print(label, label_value)
    
    def Run(self):
        self.maze.run()
    