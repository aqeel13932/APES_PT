import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num-episode', type=int, default=4000)# much more ( 1000 -> 10,000) (should be around 1 million steps)
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--models',nargs='+',default=[],type=int)
parser.add_argument("--record",default=False,help="record video file")
args = parser.parse_args()
import numpy as np
import skvideo.io
from keras.models import Model,load_model
from Settings import *
from World import *
from Agent import *
from Obstacles import *
from Foods import *
import os

def WriteInfo(Model,Type,Episode,Reward,Steps,I_C_Food,I_C_DOM,DOM_C_FOOD):
    with open('output/Models_Test.csv','a') as outp:
        #print('{},{},{},{},{},{},{},{}\n'.format(Model,Type,Episode,Reward,Steps,I_C_Food,I_C_DOM,DOM_C_FOOD))
        outp.write('{},{},{},{},{},{},{},{}\n'.format(Model,Type,Episode,Reward,Steps,I_C_Food,I_C_DOM,DOM_C_FOOD))

WriteInfo('Model','Type','Episode','Reward','Steps','I_C_Food','I_C_DOM','DOM_C_FOOD')

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

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=180,Range=-1,PdstName='ragnt',ActionMemory=args.naction)
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,Power=10,ControlRange=1,PdstName='gagnt')
    print(ragnt.ID,gagnt.ID)
    game =World(RewardsScheme=[-10,1000,-0.1],StepsLimit=1000)
    #Adding Agents in Order of Following the action
    #game.AddAgents([ragnt])
    game.AddAgents([gagnt,ragnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    return game

def TryModel(model,game,m_name,m_type):
    print('Testing Target Model')
    global AIAgent,File_Signature,TestingCounter,DAgent
    for i in range(args.num_episode):
        if(i%100)==0:
            print(m_name,m_type,i)
        game.GenerateWorld()
        game.Step()
        img = game.BuildImage()
        episode_reward=0
        observation = AIAgent.Flateoutput()
        if args.record:
            writer = skvideo.io.FFmpegWriter("output/VID_TEST/{}_{}_{}_Test.avi".format(m_name,m_type,i))
            writer.writeFrame(np.array(img*255,dtype=np.uint8))
        for t in range(1000):
            s =np.array([observation])
            q = model.predict(s, batch_size=1)
            action = np.argmax(q[0])
            AIAgent.NextAction = Settings.PossibleActions[action]
            AIAgent.AddAction(action)
            DAgent.DetectAndAstar()
            I_C_DOM = game.agents[1001].NNFeed['agentori1002'].sum() #I see Dominante 
            I_C_FOOD = game.agents[1001].NNFeed['food'].sum() # I See Food
            DOM_C_FOOD=game.agents[1002].NNFeed['food'].sum() # Dominant See Food.
            game.Step()
            if args.record:
                writer.writeFrame(np.array(game.BuildImage()*255,dtype=np.uint8))
            #writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
            observation = AIAgent.Flateoutput()
            reward = AIAgent.CurrentReward
            done = game.Terminated[0]
            #observation, reward, done, info = env.step(action)
            episode_reward += reward
            #print "reward:", reward
            if done:
                WriteInfo(m_name,m_type,i,episode_reward,t,I_C_FOOD,I_C_DOM,DOM_C_FOOD)
                break

        if args.record:
            writer.close()

game = SetupEnvironment()
AIAgent = game.agents[1001]
DAgent = game.agents[1002]
naction =  Settings.PossibleActions.shape[0]
for i in args.models:
    print('loading:',i)
    model = load_model('output/{}/MOD/model.h5'.format(i))
    target_model = load_model('output/{}/MOD/target_model.h5'.format(i))
    TryModel(model,game,i,'train')
    TryModel(model,game,i,'target')
