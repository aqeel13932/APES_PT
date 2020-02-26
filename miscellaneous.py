from APES import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import pandas.api.types as ptypes
   
def calculate_mean_window(c,vector):
    """
    This function calculate a mean window over a vector.
    Please notice that the last point might be not accurate.
    inputs:
        * c: the window length.
        * vector: 1d numpy array to calculate the mean over a specific window (c)
    Output:
        * vector: a smaller 1d array shortend to be of shape[0]/c
    """
    # see if length of vector need extra points so length%c =0
    if vector.shape[0]%c !=0:
        rem = vector.shape[0]%c
        needed = c-rem
        needed = np.zeros(c-rem)
        vector = np.concatenate([vector,needed])
    vector = np.reshape(np.array(vector),(-1,c))
    vector = np.mean(vector,axis=1)
    return vector

def training_reward_window(window,model,T='train'):
    """
    Calculate the mean of the window maximum theoretical perfect reward 
    and the mean of the window of the actual reward.
    Input:
        * window: an integer of the window to calculate its mean.
        * model: the model number.
        * T: ['train','Test'] define the type of episodes, the training ones or the testing ones.
    """
    x = pd.read_csv('output/{}/exp_details.csv'.format(model),header=None)
    x = x[x[5]==T]
    y = x[8].copy()
    y[y==0]=1000
    y[y==1]=-10
    tmp = calculate_mean_window(window,x[2])
    perfect = calculate_mean_window(window,y)
    return tmp,perfect
        
def calculate(val,index):
    """
    This function is used specifically to calculate the mean and the standard error of the mean(SEM) of
    the accuracies and losses of trying 20 models
    input: 
        * val: array (20:4:20) (models or trials:train/test loss/accuracy:episodes)
        * index: refer to which value in 2nd axis to take. (0:val_loss,1:val_acc,2:train_loss,
        3:train_accuracy)
    output: two arrays represent the mean and SEM over 20 trials with 20 episodes each (20 as an example)
    """
    mean = np.mean(val[:,index,:],axis=0)
    std = np.std(val[:,index,:],axis=0)
    return mean,std/np.sqrt(std.shape[0])

def Mean_STD_accuracies(array):
    """
    Calculate the mean and std for a 3d matrix based on the number of tries (middle axis)
    input:
        * matrix: 3d matrix ie. [8,20,2]
    output:
        * m: 2d array of shape ie.[8,2] contain mean
        * std: 2d array of shape ie.[8,2] contain standard deviation.
    """
    m = array.mean(axis=(1))
    std = array.std(axis=(1))
    return m,std

def Mean_SEM_accuracies(array):
    """
    Calculate the mean and std for a 3d matrix based on the number of tries (middle axis)
    input:
        * matrix: 3d matrix ie. [8,20,2]
    output:
        * m: 2d array of shape ie.[8,2] contain mean
        * sem: 2d array of shape ie.[8,2] contain standard error of the mean.
    """
    m = array.mean(axis=(1))
    sem = array.std(axis=(1))
    return m,sem/np.sqrt(array.shape[1])

def Get_action_sequence(Ego=False):
    """
    load a *.npz dataset that have 26400 instances in ego-centric and 31200 instances in allo-centric.
    Those files should be in NPZ folder same directory of this file.
    Input:
        * Ego: False to get allocentric, True to get Ego centric dataset.
    output:
        * action_sequence: the sequence of actions taken 
    """
    if Ego:
        unique_count=26400
    else:
        unique_count=31200
    all_simu = np.load('NPZ/in_out_{}_seq_EGO_{}.npz'.format(unique_count,Ego))
    action_sequence = all_simu['action_sequence']
    return action_sequence
    
def Get_dataset(Ego=False):
    """
    load a *.npz dataset that have 26400 instances in ego-centric and 31200 instances in allo-centric.
    Those files should be in NPZ folder same directory of this file.
    Input:
        * Ego: False to get allocentric, True to get Ego centric dataset.
    output:
        * cnn_input:(#samples,13,13,4) or (#samples,11,21,3)
        * rest_input: (#samples,8) or (#samples,4)
        * y: metric (#samples,1)
        * conv_size: (13,13,4,) or (11,21,3,) to prepare cnn_input branch in the network
        *rest_size: (8,) or (4,) to prepare rest_input branch in the network
        *naction: 5
    """
    if Ego:
        unique_count=26400
    else:
        unique_count=31200
    all_simu = np.load('NPZ/in_out_{}_seq_EGO_{}.npz'.format(unique_count,Ego))

    data = all_simu['input_target']
    

    if Ego:
        cnn_input = data[:,:693]
        rest_input = data[:,693:697]
        y = data[:,697]
        cnn_input = cnn_input.reshape((data.shape[0],11,21,3))
        rest_input = rest_input.reshape((data.shape[0],4))

        conv_size=(11,21,3,)
        rest_size=(4,)

    else:
        cnn_input = data[:,:676]
        rest_input = data[:,676:684]
        y = data[:,684]
        cnn_input = cnn_input.reshape((data.shape[0],13,13,4))
        rest_input = rest_input.reshape((data.shape[0],8))

        conv_size=(13,13,4,)
        rest_size=(8,)
    
    y = y.reshape((data.shape[0],1))
    naction =  Settings.PossibleActions.shape[0]
    return cnn_input,rest_input,y,conv_size,rest_size,naction

def train_and_test(X, y,tries=1):
    """
    This function is used to decode if X can decode y. In PT case can the input find if the food is seen or not seen by dominant based on X ? 
    Input:
        * X: the input to the decoder. 
        * y: 1d array of the target.
        * tries: how many times to retrain 
    Output:
        * training accuracy.
        * testing accuracy. 
    """
    np.random.seed(1)
    output = np.zeros((tries,2))
    for i in range(tries):

        # split into training and test episodes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

            # train linear classifier
        clf = LinearDiscriminantAnalysis()
        #clf = MLPClassifier(hidden_layer_sizes=(), solver='adam',random_state=0)
        clf.fit(X_train, y_train)
        output[i] = (clf.score(X_train, y_train), clf.score(X_test, y_test))
        # calculate training and test accuracy
    return output

def Decode_Models_Activations(tries=1):
    """
    This function decode models 1440,1358,1336,1420 activations to predict if dominant can or can not see the food.
    output:
        * accuracies: dictionary where the dictionary keys are model and the value is a numpy array of shape (8,2).
        (8:layers[input,flatten,merge,FC_1,FC_2,LSTM,FC_5,output],2:[training accuracy,validation accuracy])
        * models_description: a dictionary with the models number and a small description.
    """
    models_description={1336:'Allo-Allo',1440:'Ego-Allo',1420:'Allo-Ego',1358:'Ego-Ego'}
    accuracies={}
    for ego in [True,False]:
        cnn_input,rest_input,y,conv_size,rest_size,naction = Get_dataset(Ego=ego)
        cnn_input = cnn_input.reshape((cnn_input.shape[0],-1))
        X = np.concatenate([cnn_input,rest_input],axis=1)
        input_accuracy = train_and_test(X,y.ravel(),tries)
        if ego:
            models = [1440,1358]
        else:
            models = [1336,1420]
        for mod in models:
            print('\n Working on decoder for model {}:'.format(mod),end="")
            accuracies[mod]=np.zeros((8,tries,2))
            accuracies[mod][0] = input_accuracy
            i=1
            input_data = np.load('NPZ/activations_model_{}_ego_{}.npz'.format(mod,ego))
            for file_name in input_data.files:
                print(file_name,end=" ")
                accuracies[mod][i]=train_and_test(input_data[file_name],y.ravel(),tries)
                i+=1
    return accuracies,models_description

def Get_metric_and_action_sequence(Ego=False,reverse=False):
    """
    loabecause I would like the kitchen not take much space d a *.npz dataset that have 26400 instances in ego-centric and 31200 instances in allo-centric.
    Those files should be in NPZ folder same directory of this file.
    Input:
        * Ego: False to get allocentric, True to get Ego centric dataset.
        * Reverse: False to get same type of actions as vision
    output:
        * y: metric (#samples,1) when 1 the agent should not eat the food, when 0 the agent should eat it.
        * action_sequence: (#samples,100) represent the selected action by agent in every time step.
    """
    file_string = 'NPZ/in_out_'
    if Ego:
        file_string+= '26400_seq_EGO_True'
    else:
        file_string+= '31200_seq_EGO_False'
    if reverse:
        file_string+= '_reversed_True.npz'
    else:
        file_string+= '.npz'
    all_simu = np.load(file_string)    
    data = all_simu['input_target']
    
    if Ego:
        cnn_input = data[:,:693]
        y = data[:,697]

    else:
        cnn_input = data[:,:676]
        y = data[:,684]
    
    y = y.reshape((data.shape[0],1))
    return y,all_simu['action_sequence']

def Get_Model_Behavior(Ego=False,reverse=False):
    """
    Get the model behavior over all the possible cases.
    Input:
        * Ego: [True,False(default)] if we want egocentric vision or allocentric visoin.
        * reverse: [True,False (default)]To use same vision action (ego action with ego vision) or reverse
        the actions to the other vision.
    output:
        * avoid_i: int, number of cases where agent avoided unseen food( not good)
        * avoid_l: int, number of cases where agent avoided seen food (good)
        * ate_l: int, number of cases where agent ate unseen food (good)
        * ate_i: int, number of cases where agent ate seen food (not good)
    """
    y,action_sequence = Get_metric_and_action_sequence(Ego=Ego,reverse=reverse)
    action_sequence_sorted = np.sort(action_sequence,axis=1)
    Got_Food = action_sequence_sorted[:,0]
    Got_Food[Got_Food>=0]=True
    Got_Food[Got_Food<0]=False
    Got_Food = np.logical_not(Got_Food)
    # Agent didn't get the food and dominant didn't see the food.
    avoid_i = (y[Got_Food==False]==False).sum()
    # Agent didn't get the food and dominant saw the food.
    avoid_l = (y[Got_Food==False]==True).sum()
    # Agent got the food and dominant didn't see the food.
    ate_l = (y[Got_Food==True]==False).sum()
    # Agent got the food and dominant saw the food.
    ate_i = (y[Got_Food==True]==True).sum()
    return avoid_i,avoid_l,ate_l,ate_i

def seperate_dataset(data):
    """Analays data and count how many times the agent ate or avoided food correctly or incorrectly.
    Input:
        * data: dataset that contain experiment details usually generated file while training.
    Output:
        * el: Food eaten legally orcorrectly by subordinate agent (dominante didn't see the food).
        * ei: Food eaten ilegally (un-correctly) by subordinate agent(dominant saw food).
        * el: Food avoided legally (correctly) by subordinate agent (dominant saw food).
        * el: Food avoided ilegally (un-correctly) by subordinate agent (dominant didn't see the food).
        """
    eaten = data[(data[2]>880)|(data[2]<-880)]
    avoided = data[(data[2]<20)&(data[2]>-15)]
    if ptypes.is_numeric_dtype(data[8].dtype):
        el = eaten[eaten[8]==0].shape[0]
        ei = eaten[eaten[8]==1].shape[0]
        ai = avoided[avoided[8]==0].shape[0]
        al = avoided[avoided[8]==1].shape[0]
    else:
        el = eaten[eaten[8]=='[0]'].shape[0]
        ei = eaten[eaten[8]=='[1]'].shape[0]
        ai = avoided[avoided[8]=='[0]'].shape[0]
        al = avoided[avoided[8]=='[1]'].shape[0]
    return el,ei,al,ai

def Process_data(data,window):
    """Returen the behavior of the subordinate agent for eating or avoiding food correctly or 
    incorrectly based on the dominant ability to observed food. 
    Input:
        * data: dataset that contain experiment details usually generated file while training.
    Output:
        * draw_data: (#datashape/window,eaten_correctly,eaten_incorretly,
        avoided_correctly,avoided incorrectly)
        """
    times = int(data.shape[0]/window)
    draw_data = np.zeros((times+1,4))
    for i in range(times+1):
        draw_data[i] = seperate_dataset(data[i*window:(i+1)*window])
    return draw_data
