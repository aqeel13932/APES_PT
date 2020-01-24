from APES import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def calculate(val,index):
    """
    This function is used specifically to calculate the mean and the standard error of the mean(SEM) of
    the accuracies and losses of trying 20 models
    input: array (20:4:20) (models or trials:train/test loss/accuracy:episodes)
    output: two arrays represent the mean and SEM over 20 trials with 20 episodes each (20 as an example)
    """
    mean = np.mean(val[:,index,:],axis=0)
    std = np.std(val[:,index,:],axis=0)
    return mean,std/np.sqrt(std.shape[0])

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
    action_sequence = all_simu['action_sequence']

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

def train_and_test(X, y):
    """
    This function is used to decode if X can decode y. In PT case can the input find if the food is seen or not seen by dominant based on X ? 
    Input:
        * X: the input to the decoder. 
        * y: 1d array of the target.
    Output:
        * training accuracy.
        * testing accuracy. 
    """
    
    # split into training and test episodes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

    # train linear classifier
    clf = MLPClassifier(hidden_layer_sizes=(), solver='adam')
    clf.fit(X_train, y_train)

    # calculate training and test accuracy
    return clf.score(X_train, y_train), clf.score(X_test, y_test)