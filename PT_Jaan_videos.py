import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',default=[],type=int)
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--cnn',type=bool,default=False)
args = parser.parse_args()
if args.cnn:
    print('using CNN Model')
else:
    print('Non CNN Model')
msgs=[',should go',',should not go']
msg='Env:{}'
#FINAL TEST CASES (USED IN THE PAPER).
#'''
preferences={
#Group 1 (Dominant positions) Original
2:{'sub':(2,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
5:{'sub':(2,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
6:{'sub':(2,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
9:{'sub':(2,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
15:{'sub':(2,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'}
}
#'''
import numpy as np
np.random.seed(4917)
from keras.models import load_model
import skvideo.io
from time import time
from Settings import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from Environments import CreateEnvironment

def AddTextToImage(img,action,AgentView=0):
    img = np.array(img*255,dtype=np.uint8)
    img = Image.fromarray(img)
    #img = Image.fromarray(game.BuildImage())
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("LiberationSans-Bold.ttf", 12)
    # draw.text((x, y),"Sample Text",(r,g,b))
    if AgentView:
        draw.text((0, 0),"Action:{}".format(action),(255,0,0),font=font)
    else:
        draw.text((0, 0),"Action:{}".format(action),(0,0,0),font=font)
    return img

#### Load the model
for i in args.models:
    if not os.path.exists('Final_Results/{}'.format('Jaan')):
        os.makedirs('Final_Results/{}'.format('Jaan'))
    train = load_model('output/{}/MOD/model.h5'.format(i))
    #target = load_model('output/{}/MOD/target_model.h5'.format(i))
    models={'train':train}#,'target':target}
    #models={'target':target}
    for mod in models:
        np.random.seed(1337)
        model = models[mod]
        #domagnt=[True,False]
        domagnt=[False]
        for dom in domagnt:
            #for env in range(1,13):
            for env in preferences:
                counter = env+(env-1)*2
                preferences[env]['mesg']=preferences[env]['mesg'].format(env)
                game = CreateEnvironment(preferences[env],args.naction)
                print(game.agents[1002].Direction,env)
                DAgent,AIAgent = [game.agents[x] for x in game.agents]
                AIAgent = game.agents[1001]
                #print('D:{},AI:{}'.format(DAgent.ID,AIAgent.ID))
                #AIAgent = [game.agents[x] for x in game.agents][0]
                TestingCounter=0
                TestingCounter+=1
                writer =skvideo.io.FFmpegWriter("Final_Results/{}/ENV{}.avi".format('Jaan',env))
                writer2 =skvideo.io.FFmpegWriter("Final_Results/{}/ENV{}_AG.avi".format('Jaan',env))
                writer3 =skvideo.io.FFmpegWriter("Final_Results/{}/ENV{}_DOM.avi".format('Jaan',env))
                #game.GenerateWorld()
                img = game.BuildImage()
                game.Step()
                #plt.imsave('Final_Results/{}/ENV_{}.png'.format(i,env,TestingCounter),img)
                Start = time.time()
                episode_reward=0
                writer.writeFrame(AddTextToImage(game.BuildImage(),AIAgent.NextAction,0))
                writer2.writeFrame(AddTextToImage(game.AgentViewPoint(AIAgent.ID),AIAgent.NextAction,1))
                writer3.writeFrame(AddTextToImage(game.AgentViewPoint(DAgent.ID),'',1))
                for t in range(1000):
                    if args.cnn:
                        cnn,rest = AIAgent.Convlutional_output()
                        q = model.predict([cnn[None,:],rest[None,:]], batch_size=1)
                    else:
                        #AIAgent.NNFeed['agentori1002'].fill(0)
                        observation = AIAgent.Flateoutput()
                        s =np.array([observation])
                        q = model.predict(s, batch_size=1)
                    action = np.argmax(q[0])
                    AIAgent.NextAction = Settings.PossibleActions[action]
                    if dom:
                        DAgent.DetectAndAstar()
                    if (t>20)&(env>=9):
                        DAgent.NextAction=Settings.PossibleActions[3]
                    game.Step()
                    if args.cnn:
                        cnn,rest = AIAgent.Convlutional_output()
                    else:
                        observation = AIAgent.Flateoutput()
                    reward = AIAgent.CurrentReward
                    #print(reward)
                    done = game.Terminated[0]
                    #observation, reward, done, info = env.step(action)
                    episode_reward += reward
                    writer.writeFrame(AddTextToImage(game.BuildImage(),'{},TR:{}'.format(AIAgent.NextAction,episode_reward),0))
                    writer2.writeFrame(AddTextToImage(game.AgentViewPoint(AIAgent.ID),'{},TR:{}'.format(AIAgent.NextAction,episode_reward),1))
                    writer3.writeFrame(AddTextToImage(game.AgentViewPoint(DAgent.ID),'{}'.format(DAgent.NextAction),1))
                    if done:
                        break

                writer.close()
                writer2.close()
                writer3.close()
                Start = time.time()-Start
