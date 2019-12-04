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
parser.add_argument('--optimizer_lr', type=float, default=0.001)#could be used later priority 4.5
parser.add_argument('--exploration', type=float, default=0.1)# priority (0.8) it should decrease over time to reach 0.001 or even 0
parser.add_argument('--vanish',type=float,default=0.75)#Decide when the exploration should stop in percentage (75%)
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='avg')# priority 2 maybe done once and stike with one 
parser.add_argument('--rwrdschem',nargs='+',default=[-10,1000,-0.1],type=float) #(calculated should be (1000 reward , -0.1 punish per step)
parser.add_argument('--svision',type=int,default=180)
parser.add_argument('--details',type=str,default='')
parser.add_argument('--train_m',type=str,default='')
parser.add_argument('--target_m',type=str,default='')
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--nconv',type=int,default=2)
parser.add_argument('--cnnfilter',nargs='+',default=[32],type=int) 
parser.add_argument('--cnnsize',nargs='+',default=[3],type=int)
args = parser.parse_args()

import numpy as np
np.random.seed(args.seed)
import skvideo.io
from keras.models import Model,load_model
from keras.layers.core import Flatten
from keras.layers import Input, Dense, Lambda, convolutional, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam,rmsprop
from keras.regularizers import l2
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
    line.append(args.nconv)
    line.append(args.cnnfilter)
    line.append(args.cnnsize)
    line.append(args.optimizer_lr)
    line.append("\""+args.details+"\"")
    return ','.join([str(x) for x in line])

line = GenerateSettingsLine()
with open ('output/features.results.out','a') as f:
    f.write('{}\n{}\n'.format(File_Signature,line))

def WriteInfo(epis,t,epis_rwrd,start,rwsc,rwprob,aiproba,eptype,trqavg,tsqavg):
    global File_Signature
    with open('output/{}/exp_details.csv'.format(File_Signature),'a') as outp:
        outp.write('{},{},{},{},{},{},{},{},{},{}\n'.format(epis,t,epis_rwrd,start,rwsc,rwprob,aiproba,eptype,trqavg,tsqavg))

def SetupEnvironment():
    Start = time()

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

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=args.svision,Range=-1,PdstName='ragnt',ActionMemory=args.naction)
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,Power=10,ControlRange=1,PdstName='gagnt')
    print(ragnt.ID,gagnt.ID)
    game =World(RewardsScheme=args.rwrdschem,StepsLimit=args.max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([ragnt])
    #game.AddAgents([gagnt,ragnt])
    game.AddObstacles([obs])
    game.AddFoods([food])
    Start = time()-Start
    print ('Taken:',Start)
    return game

def createLayers(insize,in_conv,naction):
    c = Input(shape=in_conv)
    con_process = c
    for i in range(args.nconv):
        con_process = convolutional.Conv2D(filters=args.cnnfilter[i],kernel_size=args.cnnsize[i],activation="relu",padding="same",kernel_regularizer=l2(1e-5))(con_process)

        con_process= BatchNormalization(axis=1)(con_process)

    con_process = Flatten()(con_process)
    x = Input(shape=insize)#env.observation_space.shape)
    h = merge([con_process,x],mode="concat")
    for i in range(args.layers):
      h = Dense(args.hidden_size, activation=args.activation)(h)
      if args.batch_norm and i != args.layers - 1:
        h = BatchNormalization(axis=1)(h)

    
    y = Dense(naction + 1)(h)
    if args.advantage == 'avg':
      z = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(naction,))(y)
    elif args.advantage == 'max':
      z = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(naction,))(y)
    elif args.advantage == 'naive':
      z = Lambda(lambda a: K.expand_dims(a[:,0], axis=-1) + a[:,1:], output_shape=(naction,))(y)
    else:
      assert False
    return c,x, z

def RandomWalk(game):
    rw = deepcopy(game)
    rwtc = 0
    while not rw.Terminated[0]:
        rw.agents[1001].RandomAction() 
        rw.Step()
        rwtc+=1
    return rwtc
TestingCounter=0
def TryModel(model,game):
    print('Testing Target Model')
    global AIAgent,File_Signature,TestingCounter#,DAgent
    TestingCounter+=1
    #writer = skvideo.io.FFmpegWriter("output/{}/VID/{}_Test.avi".format(File_Signature,TestingCounter))
    #writer2 = skvideo.io.FFmpegWriter("output/{}/VID/{}_TestAG.avi".format(File_Signature,TestingCounter))
    game.GenerateWorld()
    game.Step()
    img = game.BuildImage()
    rwtc =0# RandomWalk(game)
    Start = time()
    episode_reward=0
    cnn,rest = AIAgent.Convlutional_output()

    #writer.writeFrame(np.array(img*255,dtype=np.uint8))
    for t in range(args.max_timesteps):
        if np.random.random()<0.05:
            action = AIAgent.RandomAction()
        else:
            q = model.predict([cnn[None,:],rest[None,:]], batch_size=1)
            action = np.argmax(q[0])
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        #DAgent.DetectAndAstar()
        game.Step()
        #writer.writeFrame(np.array(game.BuildImage()*255,dtype=np.uint8))
        #writer2.writeFrame(np.array(game.AgentViewPoint(AIAgent.ID)*255,dtype=np.uint8))
        cnn,rest = AIAgent.Convlutional_output()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]

        #observation, reward, done, info = env.step(action)
        episode_reward += reward

        #print "reward:", reward
        if done:
            break

    #writer.close()
    #writer2.close()
    if t>=999:
        plt.imsave('output/{}/PNG/{}_Test.png'.format(File_Signature,TestingCounter),img)
    #else:
        #os.remove("output/{}/VID/{}_Test.avi".format(File_Signature,TestingCounter))
        #os.remove("output/{}/VID/{}_TestAG.avi".format(File_Signature,TestingCounter))

    Start = time()-Start
    print(t)
    WriteInfo(TestingCounter,t+1,episode_reward,Start,rwtc,'0','0','Test','0','0')

game = SetupEnvironment()

AIAgent = game.agents[1001]
#DAgent = game.agents[1002]
'''
input size :
Worldsize*(Agents Count+3)+Agents Count *4
worldsize*(Agents count +3(food,observed,obstacles)) + Agents count *4 (orintation per agent)
'''
#ishape =(Settings.WorldSize[0]*Settings.WorldSize[1]*(len(game.agents)+3)+ len(game.agents)*4,)
#ishape =(Settings.WorldSize[0]*Settings.WorldSize[1]*(2+3)+ 2*4+args.naction*5,)
conv_size=(Settings.WorldSize[0],Settings.WorldSize[1],5,)
naction =  Settings.PossibleActions.shape[0]
rest_size=(args.naction*5+8,)
if args.train_m=='':
    print('train default')
    c,x, z = createLayers(rest_size,conv_size,naction)
    #x, z = createLayers(ishape,(11,11,5),naction)
    #x, z = createLayers((,10),(11,11,5),naction)
    model = Model(inputs=[c,x], outputs=z)
    model.summary()
    optimizer = adam(lr=args.optimizer_lr) if args.optimizer=='adam' else rmsprop(lr=args.optimizer_lr)
    model.compile(optimizer=optimizer, loss='mse')
else:
    model = load_model('output/{}/MOD/model.h5'.format(args.train_m))

if args.target_m=='':
    print('test from scractch')
    c,x, z = createLayers(rest_size,conv_size,naction)

    target_model = Model(inputs=[c,x], outputs=z)
    target_model.set_weights(model.get_weights())
else:
    target_model = load_model('output/{}/MOD/target_model.h5'.format(args.target_m))
mem = Buffer(args.replay_size,conv_size,rest_size,(1,))
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
#### Recording dominant direction and position spawn area ####
dompos = np.zeros(Settings.WorldSize+(4,))
posdir = {'N':0,'S':1,'E':2,'W':3}
while progress<args.totalsteps:
    i_episode+=1
    game.GenerateWorld()
    #domcurpos= np.where(game.world==DAgent.ID)
    #domcurpos= (domcurpos[0][0],domcurpos[1][0])
    #domcurpos= domcurpos+ (posdir[DAgent.Direction],)
    #dompos[domcurpos]+=1
    #wmap = deepcopy(game.world)
    #agindx = np.where(wmap==1001)
    #agindx = (agindx[0][0],agindx[1][0])
    #findx = np.where(wmap==2001)
    #findx = (findx[0][0],findx[1][0])
    rwtc=0# = RandomWalk(game)
    rwtcprob =0# DPMP(wmap,agindx,findx,rwtc)
    #print('Random Walk needed :{} steps and probability :{}'.format(rwtc,rwtcprob))
    Start = time()
    #First Step only do the calculation of the current observations for all agents
    game.Step()
    #if (2001 in DAgent.FullEgoCentric):
    #    game.foods[2001].Energy= game.foods[2001].Energy*-1
    #else:
    #    game.foods[2001].Energy=abs(game.foods[2001].Energy)
    #print(DAgent.FullEgoCentric)
    img =game.BuildImage()
    #plt.imsave('output/{}/PNG/{}_train.png'.format(File_Signature,i_episode),img)
    #Recording Video
    episode_reward=0
    cnn,rest = AIAgent.Convlutional_output()
    #observation = AIAgent.Flateoutput()
    #print(args.exploration)
    for t in range(args.max_timesteps):
        #if t%100==0:
        #    print('Step:',t,',Episode:',i_episode)
        args.exploration = max(args.exploration-EDA,0.1)
        if np.random.random() < args.exploration:
          action =AIAgent.RandomAction()
        else:
          q = model.predict_on_batch([cnn[None,:],rest[None,:]])#, batch_size=1)
          action = np.argmax(q[0])
        #prev_ob = observation
        prev_cnn,prev_rest = cnn,rest
        AIAgent.NextAction = Settings.PossibleActions[action]
        AIAgent.AddAction(action)
        #DAgent.DetectAndAstar()
        game.Step()
        #observation = AIAgent.Flateoutput()
        cnn,rest = AIAgent.Convlutional_output()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]
        #observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print "reward:", reward
        mem.add(prev_cnn,prev_rest,np.array([action]),reward,cnn,rest,done)

        for k in range(args.train_repeat):
            #prestates,actions,rewards,poststates,terminals = mem.sample(args.batch_size)
            prest_cnn,prest_rest,actions,rewards,post_cnn,post_rest,terminals = mem.sample(args.batch_size)
            #qpre = model.predict_on_batch(prestates)
            qpre = model.predict_on_batch([prest_cnn,prest_rest])
            qpost = target_model.predict_on_batch([post_cnn,post_rest])
            for i in range(qpre.shape[0]):
                if terminals[i]:
                    qpre[i, actions[i]] = rewards[i]
                else:
                    try:
                        qpre[i, actions[i]] = rewards[i] + args.gamma * np.amax(qpost[i])
                    except Exception as ex:
                        print('qpre.shape:{},i:{},actions:{}'.format(qpre.shape,i,actions[i]))
            model.train_on_batch([prest_cnn,prest_rest], qpre)
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in range(len(weights)):
                target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
            target_model.set_weights(target_weights)

        if done:
            break
    if t>=999:

        plt.imsave('output/{}/PNG/{}_train.png'.format(File_Signature,i_episode),img)
    aiprob =0# DPMP(wmap,agindx,findx,t+1)
    Start = time()-Start
    t = t+1
    progress+=t
    
    WriteInfo(i_episode,t,episode_reward,Start,rwtc,rwtcprob,aiprob,'train',qpre.mean(),qpost.mean())
    print("Episode {} finished after {} timesteps, episode reward {} Tooks {}s, Total Progress:{}".format(i_episode, t, episode_reward,Start,progress))
    total_reward += episode_reward
    if i_episode%10==0:
        TryModel(target_model,game)
        print("Average reward per episode {}".format(total_reward /i_episode))
model.save('output/{}/MOD/model.h5'.format(File_Signature))
#save dominant distribution
np.save('output/{}/MOD/domposstats'.format(File_Signature),dompos)
target_model.save('output/{}/MOD/target_model.h5'.format(File_Signature))
TryModel(target_model,game)
