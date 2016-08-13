from Settings import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from time import time
from collections import defaultdict
class Agent:
    """This class simulate the agent behavior.
    Attributes:
        * Power: agent power (used as factor to punish other agent)
        * Range: how far the agent can see
        * VA: Vision Angle (in Degrees)
        * Direction: Determine where the agent currently looking.
        * See: Determine if other agents can see through this agent (True) or not (False)
        * ControlRange: determine the distance needed for this agent to be able to punish other agents.
        * Directions: Dictionary contain the agent image (Key:['W' or 'S' etc.],Value: Image)
        * ID: Agent ID
        * IndiciesOrderName: The name of Probablity distribution
        * VisionFields: Dictionary (Key:['W' or 'S' etc.],Value: Egocentric matrix decribe what agent can see in that direction)
        * borders: tuple (Rmin,Rmax,Cmin,Cmax) to extract the world in centric vision from EgoCentrix matrix 
        * Phases: dictionary (Key:1-3, Value: Function correspond to the phase) (used to speed up in internal operation for visionfields)
        * NextAction: array contain next action/actions to be processed in the next step of the World.
        * CurrentReward: After passing step, this variable will contain the reward the agent got from the step.
        * FullEgoCentric: numpy array contain what the agent can see in EgoCentric.
        * NNFeed: Customized output (by default a list) contain what agent need as output from the world.
        * CenterPoint: Vision Field center point"""
    def __init__(self,Fname='Pics/agent_W.jpg',Power=1,Range=-1,VisionAngle=90,ControlRange=3,See=True,PdstName=None):
        """Initialize Agent
        Args:
            * Fname: Location of the image 'Pics/agent_W.jpg' (shoud be looking to East)
            * Power: integer represent agent power, the more the stronger. (not implemented )
            * Range: integer represent Agent sight range.(how far agent can see -1 full vision) (not implemented)
            * VisionAngle: the angle of agent vision.[1,360]
            * ControlRange: The distance between agent and (food or other agents) where agent can actually make actions.
            * See: True for other agent can see through this agent , False they can't
            * PdstName:(Probability Distribution Name) The name of wanted distribution (name should be linked with Settings.IndiciesOrderTable)
            * IAteFoodID: contain the food ID that has been eaten in the last step by this agent, -1 if non eaten
        Exception:
            * ValueError Exception: when ControlRange larger than Range
            * IOError Exception : When Fname file doesn't exist
        TODO:
            * Improve the Field of Vision algorithm to prepare only in (Range) view not everything (but it shold be kept this way if we want 
            to change the view in the middle of the game so it's performance vs options availability, although it's not that huge difference in performance.)."""
        self.Power=Power # provide supporiority over agents
        self.Range=Range # vision scope
        self.VA = VisionAngle # Vision Angle
        self.Direction= np.random.choice(['W','E','N','S']) # Direction of Agent
        self.See=See

        if (Range>0 and ControlRange>Range):
            raise ValueError('ControlRange Can not be larger than Range')

        if not os.path.isfile(Fname):
            raise IOError('file:\'{}\' not found'.format(Fname))
        self.ControlRange=ControlRange
        E = Settings.ImageViewer(cv2.resize(cv2.imread(Fname),Settings.BlockSize)) #Image of Robot looking West
        W = cv2.flip(E,1) # Image of agent looking East
        M = cv2.getRotationMatrix2D((E.shape[0]/2,E.shape[1]/2),-90,1)
        N = cv2.warpAffine(E,M,E.shape[0:2]) # Image of Agent looking south
        S = cv2.warpAffine(W,M,E.shape[0:2]) # Image of Agent looking North
        self.Directions = {'W' : W ,'N':N, 'S': S,'E':E}
        visionshape = (Settings.WorldSize[0]*2-1,Settings.WorldSize[1]*2-1)
        self.ID = Settings.GetAgentID() # Get Agent ID
        self.IndiciesOrderName = PdstName

        #Field of Vision in All Directions
        self.VisionFields = {'W' : np.zeros(visionshape,dtype=np.bool) ,'N':N, 'S': S,'E':E}

        #Borders to extract centric from egocentric (Rmin,Rmax,Cmin,Cmax)
        self.borders = (0,0,0,0)
        
        #defaultdict selected for performance more info at : 
        #http://stackoverflow.com/questions/17166074/most-efficient-way-of-making-an-if-elif-elif-else-statement-when-the-else-is-don
        self.Phases =defaultdict(lambda: 4,{1:self.Phase1,2:self.Phase2,3:self.Phase3})
        self.NextAction = []
        self.CurrentReward=0
        #How the agent see the world.(All on one Layer)
        self.FullEgoCentric = np.zeros(visionshape,dtype=np.int)
        #How The Agent See the world as it's going to fed to ML model'
        self.NNFeed=[]

        #Center Point in Vision Field
        self.CenterPoint=0
        #Containt the food ID that this agent Ate.
        self.IAteFoodID=-1
        self.PrepareFieldofView()

    def DrawDirectionImage(self):
        """show an image showing all the agent direction images."""
        #plt.figure(figsize=Settings.FigureSize)
        plt.figure(figsize=(5,5))
        plt.subplot(2,2,1)
        plt.imshow(self.Directions['W'])
        plt.title('Agent looking W,ID:{}'.format(self.ID))
        plt.subplot(2,2,2)
        plt.imshow(self.Directions['E'])
        plt.title('Agent looking E,ID:{}'.format(self.ID))
        plt.subplot(2,2,3)
        plt.imshow(self.Directions['N'])
        plt.title('Agent looking N,ID:{}'.format(self.ID))
        plt.subplot(2,2,4)
        plt.imshow(self.Directions['S'])
        plt.title('Agent looking S,ID:{}'.format(self.ID))

    def GetImage(self):
        """Get Image of the agent in Current direction
        Return: 
            numpy array image"""
        return self.Directions[self.Direction]
    
    def PrepareFieldofView(self):
        """Calculate Field of View for "self" agent."""
        shape = self.VisionFields['W'].shape
        TotalLevels = Settings.WorldSize[0]
        CenterPoint = Agent.GetCenterCoords(shape)
        self.CenterPoint = CenterPoint
        #print CenterPoint, TotalLevels
        self.VisionFields['W'][CenterPoint]=1
        Start = time()
        for i in range (1,TotalLevels):
            self.updatevalues(self.VisionFields['W'][CenterPoint[0]-i:CenterPoint[0]+i+1,CenterPoint[1]-i:CenterPoint[1]+i+1],self.GetElementsCount(i))
            #print self.VisionFields['W'][CenterPoint[0]-i:CenterPoint[0]+i+1,CenterPoint[1]-i:CenterPoint[1]+i+1]
        self.VisionFields['N'] = np.rot90(np.array(self.VisionFields['W'],dtype=np.bool))
        self.VisionFields['E'] = np.rot90(np.array(self.VisionFields['N'],dtype=np.bool))
        self.VisionFields['S'] = np.rot90(np.array(self.VisionFields['E'],dtype=np.bool))
        self.VisionFields['Time'] = time()-Start

        #print self.VisionFields['W']
    def updatevalues(self,array,elementsCount):
        """Update Array boundary values depending on elementsCount which determine how many values from the boundary should be changed-
        starting from center middle point to the right.
        Args:
            array: N-diminesion array to change it's boundaries.
            elementsCount: number of elements that we need to change in the array boundary.""" 
        counter =0
        R1 =Agent.GetCenterCoords(array.shape)[0]
        C1 = array.shape[1]-1
        coords = {'R1':R1,'R2':R1,'C1':C1,'C2':C1,'Phase':1}
        array[coords['R1'],coords['C1']] = True
        
        while counter<elementsCount:
            counter +=2
            self.Phases[coords['Phase']](array,coords)

    def Phase1(self,array,coords):
        """Modify the border of the array to correspond to agent view angle.
        Args:
            * array: the array we want to apply field of view on.
            * coords: contain the parameters of C1,C2,R1,R2,Phase (more info in updatevalues fucntion)

        Phase 1.
        During this phase we start from the max column(C1,C2) and middle Row (R1,R2) 
        and start moving up and down till  
        minimum row (R1 ) , max Row (R2) then we move to phase 2"""
        coords['R1'] -=1
        coords['R2'] +=1
        array[coords['R1'],coords['C1']] = True
        array[coords['R2'],coords['C2']] = True
        if coords['R1']==0 or coords['R2'] == array.shape[0]-1:
            coords['Phase']=2

    def Phase2(self,array,coords):
        """Modify the border of the array to correspond to agent view angle.
        Args:
            * array: the array we want to apply field of view on.
            * coords: contain the parameters of C1,C2,R1,R2,Phase (more info in updatevalues fucntion)

        Phase 2.
        During this phase we start from the max column (C1,C2) and Min,Max Rows (R1,R2) 
        and start changing (C1,C2 to minimum) till
        C1,C2 ==0 then we move to phase 3"""
        coords['C1'] -=1
        coords['C2'] -=1
        array[coords['R1'],coords['C1']] = True
        array[coords['R2'],coords['C2']] = True
        if coords['C1']==0 or coords['C2'] ==0:
            coords['Phase']=3

    def Phase3(self,array,coords):
        """Modify the border of the array to correspond to agent view angle.
        Args:
            * array: the array we want to apply field of view on.
            * coords: contain the parameters of C1,C2,R1,R2,Phase (more info in updatevalues fucntion)

        Phase 3.
        During this phase we start from the minimum columns (C1,C2) and Min,Max Rows (R1,R2) 
        and start changing (R1,R2) toward center till R1==R2 then we break (all border got covered)"""
        
        coords['R1'] +=1
        coords['R2'] -=1
        array[coords['R1'],coords['C1']] = True
        array[coords['R2'],coords['C2']] = True
        if coords['R1']==coords['R2']:
            coords['Phase']=4
    
    @staticmethod
    def GetCenterCoords(shape):
        """Function to calculate the center of the array
        Args:
            * shape: the array shape
        Return : (x,y) for the center coordinates."""
        return (shape[0]-1)/2 , (shape[1]-1)/2

    def GetElementsCount(self,Level):
        """calculate total amount of cells we can see beside the center depending on the vision angle and Level.
        Args:
            * VA: Vision angle.[0 -> 360]
            * Level: determine how far this level from the agent(1 step , 2 steps etc ... ) [1-> World Size -1]"""
        #Total Degrees Per Point : The required amount of degrees to include one point.
        denum = 8*Level
        #Avoid  ZeroDivisionError
        denum = 1 if denum<1 else denum
        #Total Degress Per Point
        TDPP = 360.0/denum
        Count = int(ceil(self.VA/TDPP))
        #print 'Range:{},Level:{},TDPP:{},COUNT:{}'.format(self.VA,Level,TDPP,Count)
        return Count
    
    def RandomAction(self,MRA=2,Replace=False):
        """Generate List of Random Actions for Agent
        Args:
            * MRA: Max Random Actions (Determine the maximum number of actions)
            * Replace: if True the actions might be repeated.
        TODO:
        * implement random.choice from predifined list of actions to do group of actions
        * modify np.random.randint(1,MRA) to (0,MRA) so it may include the action nothing"""
        nsamples = np.random.randint(0,MRA)
        choices = np.random.choice(len(Settings.PossibleActions),nsamples,replace=Replace)
        self.NextAction =  Settings.PossibleActions[choices]
    