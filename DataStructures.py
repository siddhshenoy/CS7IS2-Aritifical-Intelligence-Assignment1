import sys
from collections import deque

class Stack:
    def __init__(self) -> None:
        self.stack = []
    
    def push(self, data):
        self.stack.append(data)

    def pop(self):
        return self.stack.pop()
    
    def is_empty(self):
        return (len(self.stack) == 0)
    
    def clear_stack(self):
        self.stack.clear()

class Queue:
    def __init__(self) -> None:
        self.queue = deque()
    
    def enqueue(self, data):
        self.queue.append(data)
    
    def dequeue(self):
        return self.queue.popleft()
    
    def clear_queue(self):
        self.queue.clear()
    
    def has_data(self):
        return len(self.queue) > 0 

class Node:
    def __init__(self):
        self.children = []
        self.parent = None
        self.data = None
        self.extra_attributes = None
    
    def set_extra_attributes(self, attribs):
        self.extra_attributes = attribs

    def set_data(self, data):
        self.data = data
    
    def get_data(self):
        return self.data

    def add_child(self, child_node):
        self.children.append(child_node)
    
    def get_child_list(self):
        return self.children

    def set_parent(self, parent_node):
        self.parent = parent_node

    def get_parent(self):
        return self.parent


class Graph:
    def __init__(self) -> None:
        self.root_node = None
        self.exit_node = None           # Same as the node which you want to find?
        self.bfs_stack  = Stack()
        
    def traverse_bfs(self):
        if self.root_node is not None and self.exit_node is not None:
            self.bfs_stack.clear_stack()
            self.bfs_stack.push(self.root_node)    
        else:
            print("BFS: Either the root node or the exit node are empty!", file=sys.stderr)         # Print to the error stream
        pass
    
    def node_traverse_bfs(self):
        node = self.bfs_stack.pop()
        if node == self.exit_node:
            result_path = []
            pass
        else:
            list = node.get_child_list()
            reversed_node_list = list.reverse()
            for c_node in reversed_node_list:
                self.bfs_stack.push(c_node)
            self.node_traverse_bfs()

