from Settings import *
from World import *
from Agent import *
from Obstacles import *
from Foods import *
from time import time
import os

max_timesteps=1000
rwrdschem=[-10,1000,-0.1]
svision=180
def CreateEnvironment(preference,ActionMemory=0):
    Settings.Agents=1000
    Settings.Food=2000
    Settings.Obstacle=3000
    Start = time()

    #Add Pictures
    Settings.SetBlockSize(100)
    Settings.AddImage('Wall','Pics/wall.jpg')
    Settings.AddImage('Food','Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)

    #print(preference['mesg'])
    if preference['obs']!=(0,0):
        obs[preference['obs']] = 1
    ragnt[preference['sub']] =1
    gagnt[preference['dom']]=1
    food[preference['food']]=1
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt',ActionMemory=ActionMemory)
    gagnt = Agent(Fname='Pics/gagent.jpg',Power=10,VisionAngle=180,Range=-1,ControlRange=1,PdstName='gagnt')
        
    #Add Probabilities to Settings
    if preference['obs']!=(0,0):
        Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('food',food)

	#Create World Elements
    if preference['obs']!=(0,0):
        obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')  
    game =World(RewardsScheme=rwrdschem,StepsLimit=max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([gagnt,ragnt])
    ragnt.Direction=preference['subdir']
    gagnt.Direction=preference['domdir']
    #game.AddAgents([ragnt])
    if preference['obs']!=(0,0):
        game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    #print ('Taken:',Start)
    preference['Taken']=Start
    game.GenerateWorld()
    return game
#FINAL TEST CASES (USED IN THE PAPER).
def CreateFinalTestCases():	
    msgs=[',should go',',should not go']
    msg='Env:{}'
    preferences={
    #Group 1 (Dominant positions) Original
    1:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    2:{'sub':(2,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
    3:{'sub':(2,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    4:{'sub':(2,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    5:{'sub':(2,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    6:{'sub':(2,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},

    #Group 1-1 shifting all elements.
    101:{'sub':(2+1,1),'dom':(2+1,9),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    102:{'sub':(2+1,1),'dom':(7+1,4),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
    103:{'sub':(2-1,1),'dom':(10-1,0),'food':(3-1,4),'obs':(3+1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    104:{'sub':(2+1,1),'dom':(9+1,9),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    105:{'sub':(2+1,1),'dom':(10-1,5),'food':(3-1,4),'obs':(3-1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    106:{'sub':(2+1,1),'dom':(9+1,7),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
        
    #Group 1-2 shifting only dominant when is observable.
    201:{'sub':(2,1),'dom':(2+1,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    202:{'sub':(2,1),'dom':(7+1,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
    203:{'sub':(2,1),'dom':(10-1,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    204:{'sub':(2,1),'dom':(9+1,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    205:{'sub':(2,1),'dom':(10-1,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    206:{'sub':(2,1),'dom':(9+1,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
        
    #Group 1-3 shifting subordinate only.
    301:{'sub':(2+1,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    302:{'sub':(2+1,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
    303:{'sub':(2+1,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    304:{'sub':(2+1,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    305:{'sub':(2+1,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    306:{'sub':(2+1,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
        
    #Group 1-4 shifting only food.
    401:{'sub':(2,1),'dom':(2,9),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    402:{'sub':(2,1),'dom':(7,4),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
    403:{'sub':(2,1),'dom':(10,0),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    404:{'sub':(2,1),'dom':(9,9),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    405:{'sub':(2,1),'dom':(10,5),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
    406:{'sub':(2,1),'dom':(7,7),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
        
    #Group 2 Food, Obstacle Positions.

    7:{'sub':(2,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
    8:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    9:{'sub':(2,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    10:{'sub':(2,1),'dom':(2,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    #Group 2-1 shifting all elements.

    107:{'sub':(2+1,1),'dom':(2+1,9),'food':(6+1,5),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
    108:{'sub':(2+1,1),'dom':(2+1,9),'food':(3+1,4),'obs':(7+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    109:{'sub':(2+1,1),'dom':(2+1,9),'food':(7+1,4),'obs':(7+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    110:{'sub':(2+1,1),'dom':(2+1,9),'food':(1+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    #Group 2-2 shifting only dominant when is observable.

    207:{'sub':(2,1),'dom':(2+1,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
    208:{'sub':(2,1),'dom':(2+1,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    209:{'sub':(2,1),'dom':(2+1,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    210:{'sub':(2,1),'dom':(2+1,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    #Group 2-3 shifting subordinate only.

    307:{'sub':(2+1,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
    308:{'sub':(2+1,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    309:{'sub':(2+1,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    310:{'sub':(2+1,1),'dom':(2,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    #Group 2-4 shifting only food.

    407:{'sub':(2,1),'dom':(2,9),'food':(6+1,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
    408:{'sub':(2,1),'dom':(2,9),'food':(3+1,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    409:{'sub':(2,1),'dom':(2,9),'food':(7+1,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
    410:{'sub':(2,1),'dom':(2,9),'food':(1+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'}
    }
    return preferences

