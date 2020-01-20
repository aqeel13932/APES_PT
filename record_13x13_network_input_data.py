#2019/11/25 10:27 This file is the latest, you can launch ego centric and allocentric experiements from same file also you can launch all 3 levels.
#2019/12/12 19:11 This file updated the size of allo-centric map to 13x13, removed myorientation layer from ego-centric map, adpated probability matrices to work with it.
#2020/01/11 7:47 This file upate to remove any training trace. It'll be recording only.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=21)# much more ( 1000 -> 10,000) (should be around 1 million steps)
parser.add_argument('--rwrdschem',nargs='+',default=[-10,1000,-0.1],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=180)
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--Ego', action="store_true", default=False)
parser.add_argument('--Level',type=int,default=1)
args = parser.parse_args()

import numpy as np
import skvideo.io
from APES import *
from time import time
import os


def New_Reward_Function(agents,foods,rwrdschem,world,AES,Terminated):
    """Calculate All agents rewards
    Args:
        * agents: dictionary of agents contain all agents by ID
        * foods: dictionary of all foods
        * rwrdschem: Reward Schema (More info in World __init__)
        * world: World Map
        * AES: one element array
    TODO:
        * copy this function to class or __init__ documentation as example of how to build customer reward function
        * Assign Reward To Agents
        * Impelent the Food Reward Part depending on the decision of who take the food reward if two 
          agent exist in food range in same time
        * Change All Ranges to .ControlRange not (-1) it's -1 only for testing purpuse
        * Change Punish per step to not punish when agent do nothing"""
    def ResetagentReward(ID):
        #Punish for step 
        agents[ID].CurrentReward= rwrdschem[2] # -1 # rwrdschem[2] if len(agents[ID].NextAction)>0 else 0

    for x in agents:
        ResetagentReward(x)

    AvailableFoods = world[(world>2000)&(world<=3000)]
    if len(AvailableFoods)==0:
        AES[0]-=1
        Terminated[0]= True if AES[0]<=0 else Terminated[0]
    for ID in agents.keys():
        if agents[ID].IAteFoodID >-1:
            agents[ID].CurrentReward+= foods[agents[ID].IAteFoodID].Energy* rwrdschem[1]
        agntcenter = World._GetElementCoords(ID,agents[ID].FullEgoCentric)
        aborder = World._GetVisionBorders(agntcenter,agents[ID].ControlRange,agents[ID].FullEgoCentric.shape)
        
def SetupEnvironment():
    Start = time()
    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','APES/Pics/wall.jpg')
    Settings.AddImage('Food','APES/Pics/food.jpg')
    #Specify World Size
    if args.Ego: 
        Settings.WorldSize=(11,11)
    #If the map is allocentric, we use larger word to compensate for the extra input in the ego-centric.
    else:
        Settings.WorldSize=(13,13)

    #Create Probabilities
    red_Ag_PM = np.zeros(Settings.WorldSize)
    blue_Ag_PM = np.zeros(Settings.WorldSize)
    food_PM = np.zeros(Settings.WorldSize)
    blue_Ag_PM[:,0] =1
    if args.Level==1:
        if args.Ego:
            red_Ag_PM[2,4]=1
            food_PM[5,5] = 1
        else:
            red_Ag_PM[2,3]=1
            food_PM[6,5] = 1
    elif args.Level==2:
        if args.Ego:
            red_Ag_PM[5,5]=1
            food_PM[3:8,3:8] = 1
            food_PM[5,5]=0
        else:
            red_Ag_PM[6,5]=1
            food_PM[4:9,3:8] = 1
            food_PM[6,5]=0
    elif args.Level==3:
        if args.Ego:
            red_Ag_PM[3:8,3:8]=1
            food_PM[3:8,3:8] = 1
        else:
            red_Ag_PM[4:9,3:8]=1
            food_PM[4:9,3:8] = 1
            
    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('red_Ag_PM',red_Ag_PM)
    Settings.AddProbabilityDistribution('blue_Ag_PM',blue_Ag_PM)
    Settings.AddProbabilityDistribution('food_PM',food_PM)
    #Create World Elements
    food = Foods('Food',PdstName='food_PM')

    blue_Ag = Agent(Fname='APES/Pics/blue.jpg',
                    Power=3,
                    VisionAngle=args.svision,Range=-1,
                    PdstName='blue_Ag_PM',
                    ActionMemory=args.naction,
                   EgoCentric=args.Ego)
    red_Ag = Agent(Fname='APES/Pics/red.jpg',
                   VisionAngle=180,Range=-1,
                   Power=10,
                   ControlRange=1,
                   PdstName='red_Ag_PM')
    print(blue_Ag.ID,red_Ag.ID)
    game=World(RewardsScheme=args.rwrdschem,StepsLimit=100,RewardFunction=New_Reward_Function)
    #Agents added first has priority of executing there actions first.
    #game.AddAgents([ragnt])
    game.AddAgents([red_Ag,blue_Ag])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game



TestingCounter=0
game = SetupEnvironment()

AIAgent = game.agents[1001]
DAgent = game.agents[1002]
if args.Ego:
    cnn =np.zeros((args.episodes,Settings.WorldSize[0],Settings.WorldSize[1]*2-1,3,),dtype=int8)
    rest =np.zeros((args.episodes,args.naction*5+4,),dtype=int8)
else:
    cnn =np.zeros((args.episodes,Settings.WorldSize[0],Settings.WorldSize[1] ,4,),dtype=int8)
    rest =np.zeros((args.episodes,args.naction*5+8,),dtype=int8)
    
dom_see_food = np.zeros((args.episodes,1),dtype=int8)

for i in range(args.episodes):
    game.GenerateWorld()
    AIAgent.Direction='E'
    game.Step()
    AIAgent.NNFeed['obstacles']=[]
    I_C_DOM = game.agents[1001].NNFeed['agentori1002'].sum() #I see Dominante 
    I_C_FOOD = game.agents[1001].NNFeed['food'].sum() # I See Food
    DOM_C_FOOD=game.agents[1002].NNFeed['food'].sum() # Dominant See Food.
    metric = I_C_DOM and I_C_FOOD and DOM_C_FOOD
    Start = time()
    episode_reward=0
    cnn[i],rest[i] = AIAgent.Convlutional_output()
    dom_see_food[i] = metric
    print('ICD:{},ICF:{},DCF:{}'.format(I_C_DOM,I_C_FOOD,DOM_C_FOOD))
    
    
np.savez('Ego{}_Level{}_EPS{}'.format(args.Ego,args.Level,args.episodes),cnn=cnn,rest=rest,dom_see_food=dom_see_food)

