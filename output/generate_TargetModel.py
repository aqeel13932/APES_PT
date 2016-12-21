import os
import argparse
import pandas as pd
from keras.models import load_model
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--ID', type=int)
args = parser.parse_args()
expnum = args.ID
allexp = pd.read_csv('Final.csv')
allexp = allexp[['ID','tau']]
tau = allexp[allexp.ID==expnum].iloc[0][1]
if expnum<=4:
    print 'experiement {} aborted.'.format(expnum)
    exit()
if int(expnum>63):
    exit()
print 'Working on Experiement:{},With Tau:{}'.format(expnum,tau)
Files =[]
for root,dirs,files in os.walk("{}/MOD/".format(expnum)):
    Files = files
total_models=len(Files)
target_model = load_model('{}/MOD/model_{}.h5'.format(expnum,1))
target_weights = target_model.get_weights()
for j in range(2,total_models):
    print j
    model = load_model('{}/MOD/model_{}.h5'.format(expnum,j))
    weights = model.get_weights()
    #for i in range(len(weights)):
    #      target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
    target_weights = tau* weights + (1-tau)*target_weights
target_model.set_weights(target_weights)
target_model.save('{}/target_model.h5'.format(expnum))
print 'model save for exp:{}'.format(expnum)
