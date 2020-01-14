#11/01/2020 This file was used to generate data (record model after learning, no training at all.
#In this file:
    # Dominant don't move.
    # Food become poisnous if 111 happen. where (1:I_C_FOOD,1:I_C_Dom,1:DOM_C_FOOD)
    # Food is nutrition if 110 happen.
    # If experiment is out of those two options we don't need it.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filesignature',type=int)
parser.add_argument('--episodes', type=int, default=21)# much more ( 1000 -> 10,000) (should be around 1 million steps)
parser.add_argument('--max_timesteps', type=int, default=100)# 1000 
parser.add_argument('--rwrdschem',nargs='+',default=[0,1000,-0.1],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=180)
parser.add_argument('--details',type=str,default='')
parser.add_argument('--train_m',type=str,default='')
parser.add_argument('--model_eps',type=str,default=None)
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--render',action="store_true",default=False)
args = parser.parse_args()

import numpy as np
import skvideo.io
from keras.models import Model,load_model
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from APES import *
from time import time
from buffer import Buffer
from copy import deepcopy
import os
#File_Signature = int(round(time()))
File_Signature = args.filesignature

def WriteInfo(epis,t,epis_rwrd,start,rwsc,eptype,trqavg,tsqavg,metric):
    global File_Signature
    with open('output/{}/exp_details.csv'.format(File_Signature),'a') as outp:
        outp.write('{},{},{},{},{},{},{},{},{}\n'.format(epis,t,epis_rwrd,start,rwsc,eptype,trqavg,tsqavg,metric))

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
    # Check Agents in Foods Range
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
    Settings.WorldSize=(11,11)
    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    red_Ag_PM = np.zeros(Settings.WorldSize)
    blue_Ag_PM = np.zeros(Settings.WorldSize)
    food_PM = np.zeros(Settings.WorldSize)
    obs[3:8,5] = 1 
    blue_Ag_PM[:,0] =1
    red_Ag_PM[3:8,3:8]=1
    food_PM[3:8,3:8] = 1
    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('red_Ag_PM',red_Ag_PM)
    Settings.AddProbabilityDistribution('blue_Ag_PM',blue_Ag_PM)
    Settings.AddProbabilityDistribution('food_PM',food_PM)
    #Create World Elements
    obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food_PM')

    blue_Ag = Agent(Fname='APES/Pics/blue.jpg',
                    Power=3,
                    VisionAngle=args.svision,Range=-1,
                    PdstName='blue_Ag_PM',
                    ActionMemory=args.naction)
    red_Ag = Agent(Fname='APES/Pics/red.jpg',
                   VisionAngle=180,Range=-1,
                   Power=10,
                   ControlRange=1,
                   PdstName='red_Ag_PM')
    print(blue_Ag.ID,red_Ag.ID)
    game=World(RewardsScheme=args.rwrdschem,StepsLimit=args.max_timesteps,RewardFunction=New_Reward_Function)
    #Agents added first has priority of executing there actions first.
    #game.AddAgents([ragnt])
    game.AddAgents([red_Ag,blue_Ag])
    #game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game


TestingCounter=0
def TryModel(model,game):
    global AIAgent,File_Signature,TestingCounter,DAgent
    TestingCounter+=1
    if args.render:
        writer = skvideo.io.FFmpegWriter("output/{}/VID/{}_Test.avi".format(File_Signature,TestingCounter))
        writer2 = skvideo.io.FFmpegWriter("output/{}/VID/{}_TestAG.avi".format(File_Signature,TestingCounter))
    game.GenerateWorld()
    AIAgent.Direction='E'
    game.Step()
    img = game.BuildImage()
    rwtc =0# RandomWalk(game)
    Start = time()
    episode_reward=0
    cnn,rest = AIAgent.Convlutional_output()
    I_C_DOM = game.agents[1001].NNFeed['agentori1002'].sum() #I see Dominante 
    I_C_FOOD = game.agents[1001].NNFeed['food'].sum() # I See Food
    DOM_C_FOOD=game.agents[1002].NNFeed['food'].sum() # Dominant See Food.
    print('ICD:{},ICF:{},DCF:{}'.format(I_C_DOM,I_C_FOOD,DOM_C_FOOD))
    metric = I_C_DOM and I_C_FOOD and DOM_C_FOOD
    if metric:
        game.foods[2001].Energy =-1
    else:
        game.foods[2001].Energy =1
    all_cnn = np.zeros(conv_size,dtype=np.int8)
    all_rest = np.zeros(rest_size,dtype=np.int8)
    if args.render:
        writer.writeFrame(np.array(img*255,dtype=np.uint8))
        writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
    for t in range(args.max_timesteps):
        all_cnn[t]=cnn
        all_rest[t]=rest
        q = model.predict([all_cnn[None,:],all_rest[None,:]], batch_size=1)
        action = np.argmax(q[0,t])

        # Only subordinate moves, dominant is static
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        game.Step()
        if args.render:
            writer.writeFrame(np.array(game.BuildImage()*255,dtype=np.uint8))
            writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
        cnn,rest = AIAgent.Convlutional_output()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]

        #observation, reward, done, info = env.step(action)
        episode_reward += reward

        #print "reward:", reward
        if done:
            break
    if args.render:
        writer.close()
        writer2.close()

    Start = time()-Start
    print(t)
    WriteInfo(TestingCounter,t+1,episode_reward,Start,rwtc,'Test','0','0',metric)

game = SetupEnvironment()

AIAgent = game.agents[1001]
DAgent = game.agents[1002]
#input size :
#Worldsize*(Agents Count+3)+Agents Count *4
#worldsize*(Agents count +3(food,observed,obstacles)) + Agents count *4 (orintation per agent)
if args.model_eps:
    model = load_model('output/{}/MOD/model_eps:{}.h5'.format(args.train_m,args.model_eps))
else:
    model = load_model('output/{}/MOD/model.h5'.format(args.train_m))
#Framse Size
fs = (Settings.WorldSize[0]*Settings.BlockSize[0],Settings.WorldSize[1]*Settings.BlockSize[1])
total_reward = 0
#Create Folder to store the output
if not os.path.exists('output/{}'.format(File_Signature)):
        os.makedirs('output/{}'.format(File_Signature))
        os.makedirs('output/{}/VID'.format(File_Signature))
        os.makedirs('output/{}/MOD'.format(File_Signature))
progress=0
i_episode=0
conv_size=(args.max_timesteps,Settings.WorldSize[0],Settings.WorldSize[1],5,)
naction =  Settings.PossibleActions.shape[0]
rest_size=(args.max_timesteps,args.naction*5+8,)
print(conv_size,naction,rest_size)
for i in range (args.episodes):
    TryModel(model,game)
