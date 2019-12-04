#In this file:
    # Dominant don't move.
    # Food become poisnous if 111 happen. where (1:I_C_FOOD,1:I_C_Dom,1:DOM_C_FOOD)
    # Food is nutrition if 110 happen.
    # If experiment is out of those two options we don't need it.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filesignature',type=int)
parser.add_argument('--batch_size', type=int, default=10)#100 ( 100, 16,32,64,128) priority 3
parser.add_argument('--seed',type=int,default=1337)#4(CH)9(JAP)17(ITAL)
parser.add_argument('--hidden_size', type=int, default=100)#priority 2
parser.add_argument('--layers', type=int, default=1) #priority : 1.9 (it should learn regardless , but the quality differ ) 
parser.add_argument('--batch_norm', action="store_true", default=False)#priority 5 , keep turned off
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--replay_size', type=int, default=100000)# try increasing later  , priority 3.1
parser.add_argument('--train_repeat', type=int, default=1)#(2^2) , priority 1
parser.add_argument('--gamma', type=float, default=0.99)# (calculated should be 0.99) (0.99)
parser.add_argument('--tau', type=float, default=0.001)# priority 0.9 (0.001 , 0.01 , 0.1) the one that work expeirment in the domain.
parser.add_argument('--totalsteps', type=int, default=1000000)# much more ( 1000 -> 10,000) (should be around 1 million steps)
parser.add_argument('--max_timesteps', type=int, default=1000)# 1000 
parser.add_argument('--activation', choices=['tanh', 'relu'], default='relu')# experiment ( relu , tanh) priority 0.7
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')# priority 4.9
#parser.add_argument('--optimizer_lr', type=float, default=0.001)#could be used later priority 4.5
parser.add_argument('--exploration', type=float, default=0.1)# priority (0.8) it should decrease over time to reach 0.001 or even 0
parser.add_argument('--vanish',type=float,default=0.75)#Decide when the exploration should stop in percentage (75%)
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='avg')# priority 2 maybe done once and stike with one 
parser.add_argument('--rwrdschem',nargs='+',default=[-10,1000,-0.1],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=180)
parser.add_argument('--details',type=str,default='')
parser.add_argument('--train_m',type=str,default='')
parser.add_argument('--naction',type=int,default=0)
args = parser.parse_args()

import numpy as np
np.random.seed(args.seed)
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
def GenerateSettingsLine():
    global args
    line = []
    line.append(args.replay_size)
    line.append(args.layers)
    line.append(args.tau)
    line.append(args.optimizer)
    line.append(args.advantage)
    line.append(args.max_timesteps)
    line.append(args.activation)
    line.append(args.batch_size)
    line.append(args.totalsteps)
    line.append(args.exploration)
    line.append(args.vanish)
    line.append(args.gamma)
    line.append(args.hidden_size)
    line.append(args.train_repeat)
    line.append(args.batch_norm)
    line.append(args.seed)
    line.append(args.rwrdschem)
    line.append(args.svision)
    line.append("\""+args.details+"\"")
    return ','.join([str(x) for x in line])

line = GenerateSettingsLine()
with open ('output/features.results.out','a') as f:
    f.write('{}\n{}\n'.format(File_Signature,line))

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
    red_Ag_PM[5,5]=1
    food_PM[3:8,3:8] = 1
    food_PM[5,5]=0
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

def createLayers(insize,naction):
    x = Input(shape=insize)#env.observation_space.shape)
    if args.batch_norm:
      h = BatchNormalization()(x)
    else:
      h = x
    for i in range(args.layers):
      h = Dense(args.hidden_size, activation=args.activation)(h)
      if args.batch_norm and i != args.layers - 1:
        h = BatchNormalization()(h)
    y = Dense(naction + 1)(h)
    if args.advantage == 'avg':
      z = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(naction,))(y)
    elif args.advantage == 'max':
      z = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(naction,))(y)
    elif args.advantage == 'naive':
      z = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:], output_shape=(naction,))(y)
    else:
      assert False

    return x, z

TestingCounter=0
def TryModel(model,game):
    #print('Testing Target Model')
    global AIAgent,File_Signature,TestingCounter,DAgent
    TestingCounter+=1
    if TestingCounter%10==0:
        writer = skvideo.io.FFmpegWriter("output/{}/VID/{}_Test.avi".format(File_Signature,TestingCounter))
        writer2 = skvideo.io.FFmpegWriter("output/{}/VID/{}_TestAG.avi".format(File_Signature,TestingCounter))
    game.GenerateWorld()
    AIAgent.Direction='E'
    game.Step()
    img = game.BuildImage()
    rwtc =0# RandomWalk(game)
    Start = time()
    episode_reward=0
    observation = AIAgent.Flateoutput()

    I_C_DOM = game.agents[1001].NNFeed['agentori1002'].sum() #I see Dominante 
    I_C_FOOD = game.agents[1001].NNFeed['food'].sum() # I See Food
    DOM_C_FOOD=game.agents[1002].NNFeed['food'].sum() # Dominant See Food.
    print('ICD:{},ICF:{},DCF:{}'.format(I_C_DOM,I_C_FOOD,DOM_C_FOOD))
    metric = I_C_DOM and I_C_FOOD and DOM_C_FOOD
    if metric:
        game.foods[2001].Energy =-1
    else:
        game.foods[2001].Energy =1

    if TestingCounter%10==0:
        writer.writeFrame(np.array(img*255,dtype=np.uint8))
        writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
    for t in range(args.max_timesteps):
        s =np.array([observation])
        q = model.predict(s, batch_size=1)
        action = np.argmax(q[0])

        # Only subordinate moves, dominant is static
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        game.Step()
        
        if TestingCounter%10==0:
            writer.writeFrame(np.array(game.BuildImage()*255,dtype=np.uint8))
            writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
        observation = AIAgent.Flateoutput()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]

        #observation, reward, done, info = env.step(action)
        episode_reward += reward

        #print "reward:", reward
        if done:
            break

    if TestingCounter%10==0:
        writer.close()
        writer2.close()
    if t>=999:
        plt.imsave('output/{}/PNG/{}_Test.png'.format(File_Signature,TestingCounter),img)
    #else:
        #os.remove("output/{}/VID/{}_Test.avi".format(File_Signature,TestingCounter))
        #os.remove("output/{}/VID/{}_TestAG.avi".format(File_Signature,TestingCounter))

    Start = time()-Start
    print(t)
    WriteInfo(TestingCounter,t+1,episode_reward,Start,rwtc,'Test','0','0',metric)

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
game.GenerateWorld()
game.Step()
naction =  Settings.PossibleActions.shape[0]

if args.train_m=='':
    print('train default')
    x, z = createLayers(ishape,naction)
    model = Model(input=x, output=z)
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    print('test from scractch')
    x, z = createLayers(ishape,naction)
    target_model = Model(input=x, output=z)
    target_model.set_weights(model.get_weights())
else:
    model = load_model('output/{}/MOD/model.h5'.format(args.train_m))
    target_model = load_model('output/{}/MOD/target_model.h5'.format(args.target_m))

mem = Buffer(args.replay_size,ishape,(1,))
#Exploration decrease amount:
EDA = args.exploration/(args.totalsteps*args.vanish)
#Framse Size
fs = (Settings.WorldSize[0]*Settings.BlockSize[0],Settings.WorldSize[1]*Settings.BlockSize[1])
total_reward = 0
#Create Folder to store the output
if not os.path.exists('output/{}'.format(File_Signature)):
        os.makedirs('output/{}'.format(File_Signature))
        os.makedirs('output/{}/PNG'.format(File_Signature))
        os.makedirs('output/{}/VID'.format(File_Signature))
        os.makedirs('output/{}/MOD'.format(File_Signature))

progress=0
i_episode=0
while progress<args.totalsteps:
    i_episode+=1
    game.GenerateWorld()
    AIAgent.Direction='E'
    rwtc=0# = RandomWalk(game)
    Start = time()
    #First Step only do the calculation of the current observations for all agents
    game.Step()
    img =game.BuildImage()
    episode_reward=0
    observation = AIAgent.Flateoutput()

    ## Watch output 
    I_C_DOM = game.agents[1001].NNFeed['agentori1002'].sum() #I see Dominante 
    I_C_FOOD = game.agents[1001].NNFeed['food'].sum() # I See Food
    DOM_C_FOOD=game.agents[1002].NNFeed['food'].sum() # Dominant See Food.
    metric = I_C_DOM and I_C_FOOD and DOM_C_FOOD
    print('ICD:{},ICF:{},DCF:{},metric:{}'.format(I_C_DOM,I_C_FOOD,DOM_C_FOOD,metric))
    if metric:
        game.foods[2001].Energy =-1
    else:
        game.foods[2001].Energy =1
    ## End of output watch

    for t in range(args.max_timesteps):
        args.exploration = max(args.exploration-EDA,0.1)
        if np.random.random() < args.exploration:
            action =AIAgent.RandomAction()
        else:
            s =np.array([observation])
            q = model.predict_on_batch(s)#, batch_size=1)
            action = np.argmax(q[0])
        prev_ob = observation

        #Only subordinate moves, dominant is static.
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        game.Step()
        observation = AIAgent.Flateoutput()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]
        #observation, reward, done, info = env.step(action)
        episode_reward += reward
        mem.add(prev_ob,np.array([action]),reward,observation,done)
        for k in range(args.train_repeat):
            prestates,actions,rewards,poststates,terminals = mem.sample(args.batch_size)
            qpre = model.predict_on_batch(prestates)
            qpost = target_model.predict_on_batch(poststates)
            for i in range(qpre.shape[0]):
                if terminals[i]:
                    qpre[i, actions[i]] = rewards[i]
                else:
                    try:
                        qpre[i, actions[i]] = rewards[i] + args.gamma * np.amax(qpost[i])
                    except Exception as ex:
                        print('qpre.shape:{},i:{},actions:{}'.format(qpre.shape,i,actions[i]))
            model.train_on_batch(prestates, qpre)
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in range(len(weights)):
                target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
            target_model.set_weights(target_weights)

        if done:
            break
    if t>=999:

        plt.imsave('output/{}/PNG/{}_train.png'.format(File_Signature,i_episode),img)
    Start = time()-Start
    t = t+1
    progress+=t
    
    WriteInfo(i_episode,t,episode_reward,Start,rwtc,'train',qpre.mean(),qpost.mean(),metric)
    print("Episode {} finished after {} timesteps, episode reward {} Tooks {}s, metric:{}, Total Progress:{}".format(i_episode, t, episode_reward,Start,metric,progress))
    total_reward += episode_reward
    if i_episode%10==0:
        TryModel(model,game)
        print("Average reward per episode {}".format(total_reward /i_episode))
model.save('output/{}/MOD/model.h5'.format(File_Signature))
target_model.save('output/{}/MOD/target_model.h5'.format(File_Signature))
TryModel(model,game)
