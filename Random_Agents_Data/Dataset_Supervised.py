import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filesignature',type=int)
args = parser.parse_args()

#from keras.models import load_model
from Settings import *
from Obstacles import *
from Agent import *
from Foods import *
from World import *
import ast
import matplotlib.pyplot as plt
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
    gagnt[:,10]=1
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

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=180,Range=-1,PdstName='ragnt',ActionMemory=4)
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,Power=10,ControlRange=1,PdstName='gagnt')
    print(ragnt.ID,gagnt.ID)
    game =World(RewardsScheme=[-10,1000,0.1],StepsLimit=1000)
    #Adding Agents in Order of Following the action
    #game.AddAgents([ragnt])
    game.AddAgents([gagnt,ragnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    return game

game = SetupEnvironment()
AIAgent = game.agents[1001]
DAgent = game.agents[1002]
x = pd.DataFrame(columns=['Food','Observed',
                          'MyPosition','OponentPosition',
                          'Obestacles','myori',
                          'opori','enemobs',
                          'nactions','FoodObserved',
                          'original_map','game_no'])

finalized = pd.DataFrame(columns=['input','enemobs','FoodObserved','original_map','game_no'])
DAgent,AIAgent = [game.agents[x] for x in game.agents]
counter=0
number_of_games =10000
print('let the show begin')
for game_no in range(number_of_games):
    if counter>=50000:
        break
    game.GenerateWorld()
    for t in range(1000):
        if (counter%1000)==0:
            print(counter,game_no)
        AIAgent.RandomAction()
        DAgent.RandomAction()
        game.Step()
        reward = AIAgent.CurrentReward
        #print(reward)
        done = game.Terminated[0]
        #observation, reward, done, info = env.step(action)
        x.loc[counter] = [AIAgent.NNFeed['food'].astype(int16),AIAgent.NNFeed['observed'].astype(int16),
                AIAgent.NNFeed['mypos'].astype(int16),AIAgent.NNFeed['agentpos1002'].astype(int16),
                AIAgent.NNFeed['obstacles'].astype(int16),AIAgent.NNFeed['myori'].astype(int16),
                AIAgent.NNFeed['agentori1002'].astype(int16),DAgent.NNFeed['observed'].astype(int16),
                AIAgent.LastnAction.astype(int16),np.sum(DAgent.NNFeed['food'])>=1,
                game.world.copy(),game_no]
        finalized.loc[counter] = [AIAgent.Flateoutput(),DAgent.NNFeed['observed'].astype(int16),
                                  np.sum(DAgent.NNFeed['food'])>=1,game.world.copy(),game_no]
        counter+=1
        if done:
            break
print(x.shape)
x.to_csv('supervised_data{}.csv'.format(args.filesignature),index=False)
finalized.to_csv('supervised_data_finalized{}.csv'.format(args.filesignature),index=False)
