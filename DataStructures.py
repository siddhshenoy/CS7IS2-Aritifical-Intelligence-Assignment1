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
