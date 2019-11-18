import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filesignature',type=int)
parser.add_argument('--seed',type=int,default=1337)#4(CH)9(JAP)17(ITAL)
parser.add_argument('--totalsteps', type=int, default=1000000)# much more ( 1000 -> 10,000) (should be around 1 million steps)
parser.add_argument('--max_timesteps', type=int, default=1000)# 1000 
parser.add_argument('--rwrdschem',nargs='+',default=[-10,1000,-0.1],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=180)
parser.add_argument('--details',type=str,default='')
parser.add_argument('--train_m',type=str,default='')
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--samples',type=int,default=0)
args = parser.parse_args()

import numpy as np
np.random.seed(args.seed)
import skvideo.io
from keras.models import Model,load_model
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from PD_Map import DPMP
from Settings import *
from World import *
from Agent import *
from Obstacles import *
from Foods import *
from time import time
from copy import deepcopy
from buffer import Buffer
import os
#File_Signature = int(round(time()))
File_Signature = args.filesignature

def SetupEnvironment():
    #Add Pictures
    Settings.SetBlockSize(20)
    Settings.AddImage('Wall','Pics/wall.jpg')
    Settings.AddImage('Food','Pics/food.jpg')

    #Specify World Size
    Settings.WorldSize=(11,11)

    #Create Probabilities
    obs = np.zeros(Settings.WorldSize)
    ragnt = np.zeros(Settings.WorldSize)
    gagnt = np.zeros(Settings.WorldSize)
    food = np.zeros(Settings.WorldSize)
    obs[3:8,5] = 1 
    ragnt[:,0] =1
    gagnt[:,4:11]=1
    food[:,4:7]=1
    food[3:8,5] = 0

    #Add Probabilities to Settings
    Settings.AddProbabilityDistribution('Obs',obs)
    Settings.AddProbabilityDistribution('ragnt',ragnt)
    Settings.AddProbabilityDistribution('gagnt',gagnt)
    Settings.AddProbabilityDistribution('food',food)

    #Create World Elements
    obs = Obstacles('Wall',Shape=np.array([[1],[1],[1],[1]]),PdstName='Obs')
    food = Foods('Food',PdstName='food')

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=args.svision,Range=-1,PdstName='ragnt',ActionMemory=args.naction)
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,Power=10,ControlRange=1,PdstName='gagnt')
    print(ragnt.ID,gagnt.ID)
    game =World(RewardsScheme=args.rwrdschem,StepsLimit=args.max_timesteps)
    #Adding Agents in Order of Following the action
    #game.AddAgents([ragnt])
    game.AddAgents([gagnt,ragnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    return game

TestingCounter=0

game = SetupEnvironment()

AIAgent = game.agents[1001]
DAgent = game.agents[1002]
'''
input size :
Worldsize*(Agents Count+3)+Agents Count *4
worldsize*(Agents count +3(food,observed,obstacles)) + Agents count *4 (orintation per agent)
'''
#ishape =(Settings.WorldSize[0]*Settings.WorldSize[1]*(len(game.agents)+3)+ len(game.agents)*4,)
ishape =(Settings.WorldSize[0]*Settings.WorldSize[1]*(2+3)+ 2*4+args.naction*5,)
naction =  Settings.PossibleActions.shape[0]
model = load_model('output/{}/MOD/model.h5'.format(args.train_m))
total_reward = 0
#Create Folder to store the output
if not os.path.exists('Read_out/{}'.format(File_Signature)):
        os.makedirs('Read_out/{}'.format(File_Signature))
progress=0
i_episode=0
#### Recording dominant direction and position spawn area ####
dompos = np.zeros(Settings.WorldSize+(4,))
posdir = {'N':0,'S':1,'E':2,'W':3}
data = np.zeros((args.samples,636))
original = np.zeros((args.samples,11,11))
while progress<args.totalsteps:
    i_episode+=1
    game.GenerateWorld()
    domcurpos= np.where(game.world==DAgent.ID)
    domcurpos= (domcurpos[0][0],domcurpos[1][0])
    domcurpos= domcurpos+ (posdir[DAgent.Direction],)
    dompos[domcurpos]+=1
    game.Step()
    game.foods[2001].Energy=abs(game.foods[2001].Energy)
    episode_reward=0
    observation = AIAgent.Flateoutput()
    for t in range(args.max_timesteps):
        s =np.array([observation])
        q = model.predict_on_batch(s)#, batch_size=1)
        action = np.argmax(q[0])
        prev_ob = observation
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        DAgent.DetectAndAstar()
        index = progress+t
        if index>=data.shape[0]:
            break
        data[index] = np.concatenate([game.agents[1001].Flateoutput(),
                                  [game.agents[1001].NNFeed['agentori1002'].sum(), #I see Dominante 
                                   game.agents[1001].NNFeed['food'].sum(), # I See Food
                                   game.agents[1002].NNFeed['food'].sum()]]) # Dominant See Food.
        original[index] = game.world
        game.Step()
        observation = AIAgent.Flateoutput()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]
        episode_reward += reward
        if done:
            break
    if index>=data.shape[0]:
        break
    t = t+1
    progress+=t
    
    total_reward += episode_reward
    if i_episode%10==0:
        print("Average reward per episode {}".format(total_reward /i_episode))
#save dominant distribution
np.save('Read_out/{}/domposstats'.format(File_Signature),dompos)
np.save('Read_out/{}/data_{}.npy'.format(File_Signature,data.shape[0]),data)
np.save('Read_out/{}//original_{}.npy'.format(File_Signature,original.shape[0]),original)
