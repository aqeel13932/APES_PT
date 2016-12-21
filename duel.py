import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model,load_model
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from PD_Map import DPMP
from Settings import *
from World import *
from Agent import *
from Obstacles import *
from Foods import *
from time import time
from copy import deepcopy
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--min_train', type=int, default=10)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--max_timesteps', type=int, default=1000)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
#parser.add_argument('--optimizer_lr', type=float, default=0.001)
parser.add_argument('--exploration', type=float, default=0.1)
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='naive')
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('--rwrdschem',nargs='+',default=[-10,10,-1],type=float)
parser.add_argument('--svision',type=int,default=180)
parser.add_argument('--details',type=str,default='')
args = parser.parse_args()

File_Signature = int(round(time()))
def GenerateSettingsLine():
    global args
    line = []
    line.append(args.layers)
    line.append(args.tau)
    line.append(args.optimizer)
    line.append(args.advantage)
    line.append(args.max_timesteps)
    line.append(args.gym_record)
    line.append(args.activation)
    line.append(args.batch_size)
    line.append(args.episodes)
    line.append(args.exploration)
    line.append(args.gamma)
    line.append(args.hidden_size)
    line.append(args.train_repeat)
    line.append(args.display)
    line.append(args.min_train)
    line.append(args.batch_norm)
    line.append(args.rwrdschem)
    line.append(args.svision)
    line.append(args.details)
    return ','.join([str(x) for x in line])
line = GenerateSettingsLine()
with open ('output/features.results.out','a') as f:
    f.write('{}\n{}\n'.format(File_Signature,line))

def WriteInfo(epis,t,epis_rwrd,start,rwsc,rwprob,aiproba,eptype):
    global File_Signature
    with open('output/{}/exp_details.csv'.format(File_Signature),'a') as outp:
        outp.write('{},{},{},{},{},{},{},{}\n'.format(epis,t,epis_rwrd,start,rwsc,rwprob,aiproba,eptype))

def SetupEnvironment():
    np.random.seed(1337)
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

    ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=args.svision,Range=-1,PdstName='ragnt')
    gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=1,ControlRange=0,PdstName='gagnt')

    game =World(RewardsScheme=args.rwrdschem,StepsLimit=args.max_timesteps)
    #Adding Agents in Order of Following the action
    game.AddAgents([ragnt])#,gagnt])
    game.AddObstacles([obs])
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
      z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(naction,))(y)
    elif args.advantage == 'max':
      z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(naction,))(y)
    elif args.advantage == 'naive':
      z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:], output_shape=(naction,))(y)
    else:
      assert False

    return x, z

def RandomWalk(game):
    rw = deepcopy(game)
    rwtc = 0
    print ('Random Walk Started')
    while not rw.Terminated[0]:
        rw.agents[1001].RandomAction() 
        rw.Step()
        rwtc+=1
    return rwtc
TestingCounter=0
def TryModel(model,game):
    print('Testing Target Model')
    global AIAgent,File_Signature,TestingCounter
    TestingCounter+=1
    game.GenerateWorld()
    rwtc = RandomWalk(game)
    Start = time()
    episode_reward=0
    observation = AIAgent.Flateoutput()

    for t in range(args.max_timesteps):
        #img = game.BuildImage()
        #out.write(cv2.cvtColor(np.array(img*255,dtype=np.uint8),cv2.COLOR_RGB2BGR))
        s =np.array([observation])
        q = model.predict(s, batch_size=1)
        action = np.argmax(q[0])
        AIAgent.NextAction = Settings.PossibleActions[action]
        game.Step()
        observation = AIAgent.Flateoutput()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]
        #observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print "reward:", reward

        if done:
            break
    Start = time()-Start

    WriteInfo(TestingCounter,t+1,episode_reward,Start,rwtc,'0','0','Test')

game = SetupEnvironment()
AIAgent = game.agents[1001]
'''
input size :
Worldsize*(Agents Count+3)+Agents Count *4
worldsize*(Agents count +3(food,observed,obstacles)) + Agents count *4 (orintation per agent)
'''
ishape =(Settings.WorldSize[0]*Settings.WorldSize[1]*(len(game.agents)+3)+ len(game.agents)*4,)
game.GenerateWorld()
game.Step()
naction =  Settings.PossibleActions.shape[0]
x, z = createLayers(ishape,naction)
model = Model(input=x, output=z)
model.summary()
model.compile(optimizer='adam', loss='mse')

x, z = createLayers(ishape,naction)
target_model = Model(input=x, output=z)
target_model.set_weights(model.get_weights())
prestates = []
actions = []
rewards = []
poststates = []
terminals = []
#Framse Size
fs = (Settings.WorldSize[0]*Settings.BlockSize[0],Settings.WorldSize[1]*Settings.BlockSize[1])
#Video Encoding
total_reward = 0
#Create Folder to store the output
if not os.path.exists('output/{}'.format(File_Signature)):
        os.makedirs('output/{}'.format(File_Signature))
        os.makedirs('output/{}/PNG'.format(File_Signature))
        os.makedirs('output/{}/VID'.format(File_Signature))
        os.makedirs('output/{}/MOD'.format(File_Signature))

for i_episode in range(args.episodes):
    game.GenerateWorld()
    wmap = deepcopy(game.world)
    agindx = np.where(wmap==1001)
    agindx = (agindx[0][0],agindx[1][0])
    findx = np.where(wmap==2001)
    findx = (findx[0][0],findx[1][0])
    rwtc = RandomWalk(game)
    rwtcprob =0# DPMP(wmap,agindx,findx,rwtc)
    print('Random Walk needed :{} steps and probability :{}'.format(rwtc,rwtcprob))
    Start = time()
    #First Step only do the calculation of the current observations for all agents
    game.Step()
    #Recording Video
    img =game.BuildImage()
    plt.imsave('output/{}/PNG/{}.png'.format(File_Signature,i_episode+1),img)
    episode_reward=0
    observation = AIAgent.Flateoutput()
    for t in range(args.max_timesteps):
        if np.random.random() < args.exploration:
          action =AIAgent.RandomAction()
        else:
          s =np.array([observation])
          q = model.predict(s, batch_size=1)
          #print "q:", q
          action = np.argmax(q[0])
        #print "action:", action
        prestates.append(observation)
        actions.append(action)
        AIAgent.NextAction = Settings.PossibleActions[action]
        game.Step()
        observation = AIAgent.Flateoutput()
        reward = AIAgent.CurrentReward
        done = game.Terminated[0]
        #observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print "reward:", reward

        rewards.append(reward)
        poststates.append(observation)
        terminals.append(done)
        if len(prestates) > args.min_train:
          for k in range(args.train_repeat):
            if len(prestates) > args.batch_size:
              indexes = np.random.choice(len(prestates), size=args.batch_size)
            else:
              indexes = range(len(prestates))
            qpre = model.predict(np.array(prestates)[indexes])
            qpost = model.predict(np.array(poststates)[indexes])
            for i in range(len(indexes)):
              if terminals[indexes[i]]:
                  qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
              else:
                  try:
                      qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + args.gamma * np.amax(qpost[i])
                  except Exception as ex:
                      print('qpre.shape:{},i:{},actions:{}'.format(qpre.shape,i,actions[indexes[i]]))
            model.train_on_batch(np.array(prestates)[indexes], qpre)
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in range(len(weights)):
              target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
            target_model.set_weights(target_weights)

        if done:
            break
    aiprob =0# DPMP(wmap,agindx,findx,t+1)
    Start = time()-Start
    WriteInfo(i_episode+1,t+1,episode_reward,Start,rwtc,rwtcprob,aiprob,'train')
    print("Episode {} finished after {} timesteps, episode reward {} Tooks {}s".format(i_episode + 1, t + 1, episode_reward,Start))
    total_reward += episode_reward
    model.save('output/{}/MOD/model_{}.h5'.format(File_Signature,i_episode+1))
    TryModel(target_model,game)
print("Average reward per episode {}".format(total_reward / args.episodes))
model.save('output/{}/model.h5'.format(File_Signature))
target_model.save('output/{}/target_model.h5'.format(File_Signature))
