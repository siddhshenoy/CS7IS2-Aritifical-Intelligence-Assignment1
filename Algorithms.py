from abc import ABC, abstractmethod
from DataStructures import Queue, Stack
from pyamaze import COLOR
from queue import PriorityQueue
from enum import Enum
import random
import pprint
import math
import time
import cProfile

Infinity = float("inf")

pp = pprint.PrettyPrinter(depth=6,indent=4) 

def SquaredDistance(node1, node2):
    return math.sqrt(math.pow((node1[0] - node2[0]),2.0) + math.pow(abs(node1[1] - node2[1]), 2.0))    


class Solver(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.StartTime = None
        self.EndTime = None
        self.TimeDifference = None
        self.startNode = None
        self.queue = Queue()
        self.stack = Stack()
        self.exploredList = []
        #self.DirectionList = ['N', 'W', 'S', 'E']
        self.DirectionList = ['N', 'E', 'W', 'S']
        self.DirectionAdditionList = dict()
        self.FinalPath = dict()
        self.BacktrackPath = dict()
        self.DirectionAdditionList["N"] = (-1, 0)
        self.DirectionAdditionList["W"] = ( 0,-1)
        self.DirectionAdditionList["S"] = ( 1, 0)
        self.DirectionAdditionList["E"] = ( 0, 1)
        self.SolverShape = "square"
        self.SolverColor = COLOR.cyan
        self.SolverFilled = False
        self.SolverName = "Solver"
        self.searchPath = []

    @abstractmethod
    def Solve(self, node_data):
        pass
    
    def SetSolverShape(self, shape):
        self.SolverShape = shape
    def GetSolverShape(self):
        return self.SolverShape
    def GetFinalPath(self):
        return self.FinalPath
    
    def SetSolverColor(self, color):
        self.SolverColor = color
    
    def SetSolverFilled(self, filled):
        self.SolverFilled = filled

    def GetSolverFilled(self):
        return self.SolverFilled

    def GetSolverColor(self):
        return self.SolverColor
    def GetNumberOfSteps(self):
        return len(self.FinalPath) + 1
    def SetName(self, name):
        self.SolverName = name
    
    def GetName(self):
        return self.SolverName
    def GetNextNode(self, node, direction):
        return (
            node[0] + self.DirectionAdditionList[direction][0],
            node[1] + self.DirectionAdditionList[direction][1]
        )
    def GetTimeDifference(self):
        return self.TimeDifference
    
    def GetFinalPathLength(self):
        return len(self.FinalPath) + 1
    def GetSearchPathLength(self):
        return len(self.searchPath) + 1
    
    def IsNodeExplored(self, Node):
        return Node in self.exploredList

class BFS(Solver):
    def __init__(self) -> None:
        super().__init__()

    def Solve(self, maze, goal_node):
        self.StartTime = time.time()
        self.startNode = (maze.rows, maze.cols)
        self.queue.enqueue(self.startNode)
        self.exploredList.append(self.startNode)
        while self.queue.has_data():
            CurrentNode = self.queue.dequeue()
            self.searchPath.append(CurrentNode)
            if CurrentNode != goal_node:
                for Direction in self.DirectionList:
                    if maze.maze_map[CurrentNode][Direction] == True:
                        _nn = self.GetNextNode(CurrentNode, Direction)
                        if self.IsNodeExplored(_nn) == False:
                            self.queue.enqueue(_nn)
                            self.exploredList.append(_nn)
                            self.BacktrackPath[_nn] = CurrentNode
            else:
                break
        CurrentNode = goal_node
        while CurrentNode != self.startNode:
            self.FinalPath[self.BacktrackPath[CurrentNode]] = CurrentNode
            CurrentNode = self.BacktrackPath[CurrentNode]
        self.EndTime = time.time()
        self.TimeDifference = self.EndTime - self.StartTime

class DFS(Solver):
    def __init__(self) -> None:
        super().__init__()

    def Solve(self, maze, goal_node):
        self.StartTime = time.time()
        self.startNode = (maze.rows, maze.cols)
        self.stack.push(self.startNode)
        self.exploredList.append(self.startNode)
        while not self.stack.is_empty():
            CurrentNode = self.stack.pop()
            self.searchPath.append(CurrentNode)
            if CurrentNode != goal_node:
                for Direction in self.DirectionList:
                    if maze.maze_map[CurrentNode][Direction] == True:
                        _nn = self.GetNextNode(CurrentNode, Direction)
                        if self.IsNodeExplored(_nn) == False:
                            self.stack.push(_nn)
                            self.exploredList.append(_nn)
                            self.BacktrackPath[_nn] = CurrentNode
            else:
                break
        CurrentNode = goal_node
        while CurrentNode != self.startNode:
            self.FinalPath[self.BacktrackPath[CurrentNode]] = CurrentNode
            CurrentNode = self.BacktrackPath[CurrentNode]
        self.EndTime = time.time()
        self.TimeDifference = self.EndTime - self.StartTime

class AStar(Solver):
    
    def __init__(self) -> None:
        super().__init__()
        self.scores = dict()
        self.priq = PriorityQueue()
        self.HeuristicFunction = None
    
    def SetHeuristicFunction(self, function):
        self.HeuristicFunction = function
    def InitializeScore(self, maze):
        return {_s : Infinity for _s in maze.grid}
    def Solve(self, maze, goal_node):
        self.StartTime = time.time()
        if self.HeuristicFunction != None:
            self.startNode = (maze.rows, maze.cols)
            self.priq.put(
                (
                    self.HeuristicFunction(self.startNode, goal_node),
                    self.HeuristicFunction(self.startNode, goal_node),
                    self.startNode
                )
            )
            self.scores["Weight"] = self.InitializeScore(maze)
            self.scores["Heuristic"] = self.InitializeScore(maze)
            self.scores["Weight"][self.startNode] = 0
            self.scores["Heuristic"][self.startNode] = self.HeuristicFunction(self.startNode, goal_node)

            while not self.priq.empty():
                CurrentNode = self.priq.get()
                CurrentNode = CurrentNode[2]
                self.searchPath.append(CurrentNode)
                if CurrentNode != goal_node:
                    for Direction in self.DirectionList:
                        if maze.maze_map[CurrentNode][Direction] == True:
                            #NextCell = (CurrentNode[0] + self.DirectionAdditionList[Direction][0], CurrentNode[1] + self.DirectionAdditionList[Direction][1]) 
                            _nn = self.GetNextNode(CurrentNode, Direction)
                            weight = self.scores["Weight"][CurrentNode] + 1
                            heuristic = weight + self.HeuristicFunction(_nn, goal_node)
                            if heuristic < self.scores["Heuristic"][_nn]:
                                self.BacktrackPath[_nn] = CurrentNode
                                self.scores["Weight"][_nn] = weight
                                self.scores["Heuristic"][_nn] = weight + self.HeuristicFunction(_nn, goal_node)
                                self.priq.put(
                                    (
                                        self.scores["Heuristic"][_nn],
                                        self.HeuristicFunction(_nn, goal_node),
                                        _nn
                                    )
                                )
                else:
                    break
            CurrentNode = goal_node
            while CurrentNode != self.startNode:
                self.FinalPath[self.BacktrackPath[CurrentNode]] = CurrentNode
                CurrentNode = self.BacktrackPath[CurrentNode]
        else:
            print("AStar: No heuristic function was set hence the algorithm was not run!")
        self.EndTime = time.time()
        self.TimeDifference = (self.EndTime - self.StartTime)


class MDPSolverStrategy(Enum):
    ValueIteration = 1
    PolicyIteration = 2

class MDPMetrics:
    Reward = 0
    Discount = 1
    MaximumError = 2

class BaseMDP:
    def __init__(self) -> None:
        self.Metrics = [0, 0, 0]
        self.Metrics[MDPMetrics.Reward] = -10
        self.Metrics[MDPMetrics.Discount] = 0.9
        self.Metrics[MDPMetrics.MaximumError] = 0.001
        self.RewardList = dict()
        self.ActionList = dict()
        self.UtilityValues = dict()
        self.PolicyList = dict()
        self.ExploredList = []
    
    def SetDiscount(self, discount):
        self.Metrics[MDPMetrics.Discount] = discount
    def SetReward(self, reward):
        self.Metrics[MDPMetrics.Reward] = reward
    def SetMaximumError(self, error):
        self.Metrics[MDPMetrics.MaximumError] = error
    
class MDPVI(Solver, BaseMDP):
    def __init__(self) -> None:
        super().__init__()
        self.Metrics[MDPMetrics.Reward] = -10
        self.Metrics[MDPMetrics.Discount] = 0.5
        self.Metrics[MDPMetrics.MaximumError] = 0.001
        self.pvalues = {"N" : 1, "W" : 1, "S": 1, "E": 1}

    def AssignStochasticValues(self):
        for key, value in self.ActionList.items():
            for k, v in value.items():
                value[k] = self.pvalues[k]

    def GenerateActionLists(self, maze):
        for n, d in maze.maze_map.items():
            self.ActionList[n] = dict([(k, v) for k,v in d.items() if v == 1])
        self.AssignStochasticValues()
    
    def GenerateUtilityValues(self):
        self.UtilityValues = {a: 0 for a in self.ActionList.keys()}

    def Iterate(self, maze, goal_node):
        self.UtilityValues[goal_node] = 1
        for i in range(0, 1):
            while True:
                d = 0
                for state in self.ActionList.keys():
                    if state == goal_node:
                        continue
                    _mu = float("-inf")
                    for action, p in self.ActionList[state].items():
                        for direction in action:
                            if maze.maze_map[state][direction] == True:
                                _ns = self.GetNextNode(state, direction)
                        reward = self.Metrics[MDPMetrics.Reward]
                        if _ns == goal_node:
                            reward = 10000
                        _u = (reward + p * self.Metrics[MDPMetrics.Discount] * self.UtilityValues[_ns])
                        if _u > _mu:
                            _mu = _u
                    d = self.ReCalculateDel(state, d, _mu)
                    self.UtilityValues[state] = _mu
                if d < self.Metrics[MDPMetrics.MaximumError]:
                    break 
    def ReCalculateDel(self, state, old_d, mu):
        return max(old_d, abs(mu - self.UtilityValues[state]))
    def GetNextStateReward(self, maze, state, goal):
        total_outlets = 0
        state_outlets = 0
        for direction in self.DirectionList:
            if maze.maze_map[state][direction] == True:
                state_outlets += 1
            total_outlets += 1
        return math.pow(SquaredDistance(goal, state), 2)
    def TracePath(self, currentNode, maze, goal_node):
        node = currentNode
        self.ExploredList.append(currentNode)
        should_exit = False
        while True and should_exit == False:
            node_data = {
                "BN" : None, "BNV" : None
            }
            if node != goal_node:
                for direction in self.DirectionList:
                    if maze.maze_map[node][direction] == True and self.GetNextNode(node, direction) not in self.ExploredList:
                        _next_node = self.GetNextNode(node, direction)
                        if _next_node != goal_node:
                            if node_data["BNV"] == None:
                                node_data["BN"] = _next_node
                                node_data["BNV"] = self.UtilityValues[node_data["BN"]]
                            else:
                                temp = _next_node
                                if node_data["BNV"] < self.UtilityValues[temp]:
                                    node_data["BN"] = temp
                                    node_data["BNV"] = self.UtilityValues[temp]               
                        else:
                            node_data["BN"] = _next_node
                            node_data["BNV"] = self.UtilityValues[node_data["BN"]]
                            should_exit = True
                self.ExploredList.append(node_data["BN"])
                self.FinalPath[node] = node_data["BN"]
                node = node_data["BN"]
            else:
                break
    def Solve(self, maze, goal_path):
        self.GenerateActionLists(maze)
        self.GenerateUtilityValues()
        self.StartTime = time.time()
        self.Iterate(maze, goal_path)
        self.TracePath((maze.rows, maze.cols), maze, goal_path)
        self.EndTime = time.time()
        self.TimeDifference = self.EndTime - self.StartTime
        
class MDPPI(Solver, BaseMDP):
    def __init__(self) -> None:
        super().__init__()
        self.Metrics[MDPMetrics.Reward] = -10
        self.Metrics[MDPMetrics.Discount] = 0.2
        self.Metrics[MDPMetrics.MaximumError] = 0.001
        self.pvalues = {"N" : 1, "W" : 1, "S": 1, "E": 1}
        
    def AssignStochasticValues(self):
        for key, value in self.ActionList.items():
            for k, v in value.items():
                value[k] = self.pvalues[k]
    def GenerateActionLists(self, maze):
        for n, d in maze.maze_map.items():
            self.ActionList[n] = dict([(k, v) for k,v in d.items() if v == 1])
        self.AssignStochasticValues()
        self.RewardList = {state: -25 for state in self.ActionList.keys()}
        self.PolicyList = {state: 'E' for state in self.ActionList.keys()}
    
    def GenerateUtilityValues(self):
        self.UtilityValues = {a: 0 for a in self.ActionList.keys()}

    def Iterate(self, maze, goal_node):
        self.UtilityValues[goal_node] = 10**(8)
        self.RewardList[goal_node] = 10**(8)
        policy_update = True
        while policy_update:
            policy_update = False
            for _s in self.ActionList.keys():
                if _s != goal_node:
                    _mu,_act = float("-inf"), None
                    for action, prob in self.ActionList[_s].items():
                        for direction in action:
                            if maze.maze_map[_s][direction] == True:
                                next_state = self.GetNextNode(_s, direction)

                        reward = self.RewardList[_s]
                        if next_state == goal_node:
                            reward = 10**(100)
                        _u = reward + self.Metrics[MDPMetrics.Discount] * (prob * self.UtilityValues[next_state])
                        if _u > _mu:
                            _mu = _u
                            _act = action
                        self.PolicyList[_s] = _act
                        self.UtilityValues[_s] = _mu
                        if self.PolicyList[_s] != _act:
                            policy_update = True
                            self.PolicyList[_s] = _act
                else:
                    continue

    def TracePath(self, currentNode, maze, goal_node):
        _n = currentNode
        while _n != goal_node:
            _nn = self.GetNextNode(_n, self.PolicyList[_n])
            self.FinalPath[_n] = _nn
            _n = _nn

    def Solve(self, maze, goal_path):
        self.GenerateActionLists(maze)
        self.GenerateUtilityValues()
        self.StartTime = time.time()
        self.Iterate(maze, goal_path)
        self.TracePath((maze.rows, maze.cols), maze, goal_path)
        self.EndTime = time.time()
        self.TimeDifference = self.EndTime - self.StartTime