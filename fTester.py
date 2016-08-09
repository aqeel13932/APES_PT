import numpy as np
from time import time
from Settings import *
from Agent import *
from Obstacles import *
from Foods import *
from World import * 
np.random.seed(10) 
#print 'OpenCV:{},Numpy:{} '.format(cv2.__version__ ,np.version.full_version)
start = time()
Settings.WorldSize=(15,15)
#Settings.WorldSize=  (9,9)
# Define Distribution (Some Poplular Distributions) (Same probability for all active Blocks)
CP = Settings.WorldSize[0]/2
# Upper left
UL = np.zeros(Settings.WorldSize)
UL[0:CP,0:CP]=1
#Upper right 
UR = np.zeros(Settings.WorldSize)
UR[0:CP,CP:]=1

#Bottom Left
BL = np.zeros(Settings.WorldSize)
BL[CP:,0:CP]=1

#Bottom Right
BR= np.zeros(Settings.WorldSize)
BR[CP:,CP:]=1

#Center
C  = np.zeros(Settings.WorldSize)
number = CP/2
C[CP-number:CP+number,CP-number:CP+number]=1

Settings.AddProbabilityDistribution(Name='UL',IntProbabilityDst=UL)
Settings.AddProbabilityDistribution(Name='UR',IntProbabilityDst=UR)
Settings.AddProbabilityDistribution(Name='BL',IntProbabilityDst=BL)
Settings.AddProbabilityDistribution(Name='BR',IntProbabilityDst=BR)
Settings.AddProbabilityDistribution(Name='C',IntProbabilityDst=C)

#Add Images
Settings.AddImage('Wall','Pics/wall.jpg')
Settings.AddImage('Water','Pics/water.jpg')
Settings.AddImage('Food','Pics/food.jpg')
Settings.Images[0]=np.tile(1,(Settings.BlockSize[0],Settings.BlockSize[1],3)) #Empty
Settings.Images[-1] =np.tile(0,(Settings.BlockSize[0],Settings.BlockSize[1],3)) # black or unobservable

#Define Agents 
gAgent = Agent(Fname='Pics/gagent.jpg',PdstName='UL',Range=-1)
rAgent = Agent(Fname='Pics/ragent.jpg',PdstName='UR',Range=2,ControlRange=1)

#Create Obstacles
Plusobs = Obstacles('Water',Shape=np.array([[0,1,0],[1,1,1],[0,1,0]]),PdstName='C')
Tobs = Obstacles('Wall',Shape=np.array([[1,1,1],[0,1,0],[0,1,0]]),PdstName='C')
Robs = Obstacles('Wall',Shape=np.array([1,1,1]),PdstName='C')
Cobs = Obstacles('Wall',Shape=np.array([[1],[1],[1]]),PdstName='C')
BLOCK = Obstacles('Wall',Shape=np.array([[1,1],[1,1]]),PdstName='C')
Unite = Obstacles('Wall',Shape=np.array([1]),PdstName='C')

#Create Food
hamburger = Foods(ImageKey='Food',See=False,PdstName='BR')
hamburger2 = Foods(ImageKey='Food',See=False,PdstName='BR')

ElementsInitTime = time()-start
#print 'Initialization cost:',ElementsInitTime

start = time()
t = World()
t.AddAgents([rAgent,gAgent])
t.AddFoods([hamburger,hamburger2])
t.AddObstacles([Plusobs,Tobs,Robs,Cobs,BLOCK,Unite])
# Temproray for Reproducability
Start = time()
t.GenerateWorld()
ElementsInitTime = time()-Start
#print 'World Generating Cost : ',ElementsInitTime
print gAgent.ID,gAgent.Direction,gAgent.VA
Start = time()
t._GetTotalVision(gAgent.ID)
ElementsInitTime = time()-Start
print 'Full Total Vision for Two agents cost:',ElementsInitTime