import numpy as np
from Queue import PriorityQueue
from time import time
class Node:
    def __init__(self,r,c,G,H,Fathercords):
        self.r = r
        self.c = c
        self.F = G+H
        self.G = G
        self.H = H
        self.Father=Fathercords

class Astar:
    def __init__(self,arr,start,end):
        '''
        self.arr = arr
        self.start = start
        self.end = end
        '''
        self.arr = arr
        self.start = start
        self.end = end
        self.Open=PriorityQueue()
        self.Empty = 0
        self.Closed={}
        self.visited={}
        self.Open.put((0,Node(start[0],start[1],0,0,None)))
        self.visited[(start[0],start[1])]=True
        self.GoalFound=False
    
    def Best_Next_Step(self):
        while not self.Open.empty():
            __,a = self.Open.get()
            self.Explore(a)
            self.Closed[(a.r,a.c)]= a
        p = self.end
        MoveMap={}
        if not p in self.Closed.keys():
            print 'no path exist found'
            return 
        while p!=self.start:
            MoveMap[self.Closed[p].Father]=p
            p = self.Closed[p].Father
            self.arr[p]=-10
        print self.arr
        return MoveMap
    
    def isInvalidIndex(self,r,c):
        """ Check if coordinats in range of current map or not empty.
        Args:
            * x: rows
            * y: columns
        """
        if r<0 or r>=self.arr.shape[0] or c<0 or c>=self.arr.shape[1] or (r,c) in self.visited.keys() or self.GoalFound:
            return True
        else:
            if self.arr[r,c]==-1 or self.arr[r,c]>3000:
                return True
            return False
    
    def AddtoQueue(self,ncoords,father):
        t = Node(ncoords[0],ncoords[1],father.G+1,self.Distance(ncoords,self.end),(father.r,father.c))
        if ncoords == self.end:
            self.GoalFound=True
        self.Open.put((t.F,t))
        self.visited[ncoords]=True

    def Explore(self,a):
        if not self.isInvalidIndex(a.r-1,a.c):
            self.AddtoQueue((a.r-1,a.c),a)
        if not self.isInvalidIndex(a.r+1,a.c):
            self.AddtoQueue((a.r+1,a.c),a)
        if (not self.isInvalidIndex(a.r,a.c-1)):
            self.AddtoQueue((a.r,a.c-1),a)
        if (not self.isInvalidIndex(a.r,a.c+1)):
            self.AddtoQueue((a.r,a.c+1),a)
            
            
    def Distance(self,a,b):
        return abs((a[0])-b[0])+abs((a[1])-b[1])
#s = time()
#x = Astar(np.zeros((10,10),dtype=int),(1,1),(8,8)) 
#print x.Find_Best_Root()
#print time()-s