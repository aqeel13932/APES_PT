#from APES import *
#import numpy as np
from keras.models import load_model,Model
from keras.layers import Input,convolutional,Flatten,merge,Dense
from keras.models import load_model,Model
from miscellaneous import Get_dataset
import numpy as np

def Prepare_model(mod=1336):
    """ import a model and create another one copied from it to extract activations.
    input: model number ex:1336.
    output: keras model that take input cnn,rest and outputs 7 arrays (flatten,merge,FC_1,FC_2,LSTM,FC_5+1,output)
    """
    x = load_model('output/{}/MOD/model.h5'.format(mod))
    nm = Model(inputs=[x.layers[i].input for i in [0,3]],
               outputs=[x.layers[i].output for i in [2,4,5,6,7,8,9]])
    return nm

def Generate_Activaton(Experment_Ego=True,Use_Model=1358,chunk=5000):
    """
    Generate *.npz files contain the activations of the specificed model of 7 outputs from the same model.
    Inputs:
        * Experment_Ego: True or false. True: Ego-centric, False: allo-centric
        * Use_Model: the model number. It is important that the chosen model match the vision chosen.
        * Chunk: to fit into the memory, you can optimize the chunk based on you gpu (ram).
    outputs:
        * Save an *.npz file to hard disk. activations_model_{}_ego_{}.npz'.format(Use_Model,Experment_Ego)
    """
    print("Please pay attention that this function is very memory hungry. It originally took 30GB of ram")
    cnn_input,rest_input,y,conv_size,rest_size,naction = Get_dataset(Ego=Experment_Ego)
    print(cnn_input.shape,rest_input.shape,y.shape)
    print(conv_size,naction,rest_size)
    instances= cnn_input.shape[0]
    nm = Prepare_model(Use_Model)
    if Experment_Ego:
        xx = 11
        yy = 21
        z = 3
        r = 4
    else:
        xx = 13
        yy = 13
        z = 4
        r = 8
    #dstribute data to fit the memory
    start_limit=0
    tmp=[]
    kill_flag=False
    for i in range(chunk,cnn_input.shape[0]+chunk,chunk):
        if i>cnn_input.shape[0]:
            i=cnn_input.shape[0]
            kill_flag=True
        print(start_limit,i)
        ## for ego centric 11,21,3 for allo centric 13,13,4
        inputt = np.zeros((i-start_limit,100,xx,yy,z),dtype=np.int8)
        ## for ego centric 4 for allo centric 8
        restt = np.zeros((i-start_limit,100,r),dtype=np.int8)
        inputt[:,0] = cnn_input[start_limit:i]
        restt[:,0] = rest_input[start_limit:i]
        tmp = nm.predict_on_batch([inputt,restt])
        ##Create arrays for first time only (to get the shape of layer output)
        if start_limit==0:
            activations={}
            keys=['flatten','merge','FC_1','FC_2','LSTM','FC_5+1','output']
            for j in range(7):
                activations[keys[j]] = np.zeros((instances,tmp[j].shape[2]))
        ## Use only first step information, and detch the remaining since we only moved one step only.
        for j in range(7):
            activations[keys[j]][start_limit:i]=tmp[j][:,0,:]
        if kill_flag:
            break
        start_limit=i
    ## Save the output in npz to be used later.
    np.savez('activations_model_{}_ego_{}.npz'.format(Use_Model,Experment_Ego),
             flatten=activations['flatten'],merge=activations['merge'],
             FC_1=activations['FC_1'],FC_2=activations['FC_2'],
             LSTM = activations['LSTM'],FC_5=activations['FC_5+1'],
             output=activations['output'])
    
def create_RL_layers(insize,in_conv,naction):
    """
    Create a keras model similar to reinforcement learning model but without LSTM
    Input:
        * insize: shape of orientatoins input part.
        * in_conv: the shape of the convolutional input part.
        * naction: the number of actions (output)
    Output: 
        * c: the Input layer for convolutional part.
        * x: the input layer for rest part.
        * z: the output layer.
    """
    c = Input(shape=in_conv)
    con_process = c
    con_process = convolutional.Conv2D(filters=6,kernel_size=(3,3),activation="relu",padding="same",strides=1)(con_process)
    con_process = Flatten()(con_process)
    x = Input(shape=insize)#env.observation_space.shape)
    h = merge([con_process,x],mode="concat")
    h = Dense(32, activation='tanh')(h)
    h = Dense(32, activation='tanh')(h)
    z = Dense(1, activation='sigmoid')(h)
    return c,x, z

def Supervized_test(num_exper=20,epochs=20,Ego=False,bs=512,vs=0.2):
    """
    Train a network with the reinforcement learning archeticture with the exception of 
    """
    cnn_input,rest_input,y,conv_size,rest_size,naction = Get_dataset(Ego=Ego)
    info = np.zeros((num_exper,4,epochs))
    for i in range(num_exper):
        c,x,z = create_RL_layers(rest_size,conv_size,naction)

        allo_classifier = Model(inputs=[c,x],outputs=z)

        allo_classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
        allo_classifier.summary()

        allo_history = allo_classifier.fit([cnn_input,rest_input],
                                           y,epochs=epochs,batch_size=bs,validation_split=vs)

        info[i,0,:] = allo_history.history['val_loss']
        info[i,1,:] = allo_history.history['val_acc']
        info[i,2,:] = allo_history.history['loss']
        info[i,3,:] = allo_history.history['acc']
    return info