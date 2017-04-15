from PD_Map import DPMP
from Settings import *
from World import *
from Agent import *
from Obstacles import *
from Foods import *
from time import time
import os

filesignature = 189
batch_size = 32
exploration= 0.0
tau=0.001
activation='tanh'
advantage='max'
seed=4917
totalsteps=5000 
details="A*,180,FR,E186"
hidden_size=100
layers=1
batch_norm=False
replay_size=100000
train_repeat=1
gamma=0.99
max_timesteps=1000
optimizer='adam'
vanish=0.75
rwrdschem=[-10,1000,-0.1]
svision=180
def CreateEnvironment(preference):
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

	#print(preference['mesg'])
	if preference['obs']!=(0,0):
		obs[preference['obs']] = 1
	ragnt[preference['sub']] =1
	gagnt[preference['dom']]=1
	food[preference['food']]=1
	Settings.AddProbabilityDistribution('ragnt',ragnt)
	Settings.AddProbabilityDistribution('gagnt',gagnt)
	ragnt = Agent(Fname='Pics/ragent.jpg',Power=3,VisionAngle=svision,Range=-1,PdstName='ragnt')
	gagnt = Agent(Fname='Pics/gagent.jpg',VisionAngle=180,Range=-1,ControlRange=0,PdstName='gagnt')
	ragnt.Direction=preference['subdir']
	gagnt.Direction=preference['domdir']
		
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
	#game.AddAgents([gagnt,ragnt])
	game.AddAgents([ragnt])
	if preference['obs']!=(0,0):
		game.AddObstacles([obs])
	game.AddFoods([food])
	Start = time()-Start
	#print ('Taken:',Start)
	preference['Taken']=Start
	game.GenerateWorld()
	return game
