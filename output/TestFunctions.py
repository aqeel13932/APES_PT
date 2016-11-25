import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def GetSimpleStatistics(exid):
    df = pd.read_csv('{}/exp_details.csv'.format(exid),header=None)#,delimiter=',')
    df.columns = ['Episode','Steps','Reward','Time','RWsteps','RWPD','AIPD']
    output = df.describe()
    t = df.median()
    output = output.append(pd.DataFrame([[t[0],t[1],t[2],t[3],t[4],t[5],t[6]]],columns=output.columns))
    output = output.rename({0:'median'})
    output = pd.concat([output.iloc[0:2],output.iloc[8:9],output.iloc[2:8]])
    return output

def PlotStepsVsEpisodes(exid,ax,WithRandomWalk=False,withaxis=False):
    #print exid
    df = pd.read_csv('{}/exp_details.csv'.format(exid),header=None)#,delimiter=',')
    if 1474081039<exid<1476452872:
        df.columns = ['Episode','Steps','Reward','Time']
        WithRandomWalk=False
    else:
        df.columns = ['Episode','Steps','Reward','Time','RWsteps','RWPD','AIPD']

    ax.plot(df.Episode,df.Steps,color='b',lw=1.5)
    if withaxis:
		ax.set_xlabel('Episode')
		ax.set_ylabel('Steps Count')
    ax.set_title('{}'.format(exid))
    if WithRandomWalk:
        ax.plot(df.Episode,df.RWsteps,color='r',lw=1.0)

def PlotMultipleExperiments(exidlst,WithRandomWalk=False,withaxis=False,Plts_onX=2,HorizintalDistance=0.2,PlotsHorzDist=5):
    y=1
    while Plts_onX*y<len(exidlst):
        y+=1
    fig = plt.figure(figsize=(7*Plts_onX, PlotsHorzDist*y))
    fig.subplots_adjust(hspace=HorizintalDistance)
    for i in range(len(exidlst)):
        ax = plt.subplot( y,Plts_onX,i+1)
        PlotStepsVsEpisodes(exidlst[i],ax,WithRandomWalk,withaxis)
    return plt

def totaltime(exidlst):
	output={}
	for exid in exidlst:
		df = pd.read_csv('{}/exp_details.csv'.format(exid),header=None)#,delimiter=',')
		if 1474081039<exid<1476452872:
			df.columns = ['Episode','Steps','Reward','Time']
			WithRandomWalk=False
		else:
			df.columns = ['Episode','Steps','Reward','Time','RWsteps','RWPD','AIPD']
		output[str(exid)]= df.Time.sum()
	return output
	
def GetBestPlacesFigure(exid,th=500):
    df = pd.read_csv('{}/exp_details.csv'.format(exid),header=None)#,delimiter=',')
    df.columns = ['Episode','Steps','Reward','Time','RWsteps','RWPD','AIPD']
    ax = plt.figure(figsize=(15,10))
    ax.suptitle('Episode vs Steps', fontsize=14, fontweight='bold')
    a=0
    b=0
    start=False
    maxv = (0,0,0)
    for i in range(df.shape[0]):
        if (df.iloc[i].Steps<th):
            plt.axvspan(i+0.5,i+1.5,color='y',alpha=0.5,lw=0)
            if not start:
                a=i
                b=i
                start=True
                continue
            b+=1
        else:
            start=False
            if (a==b):
                continue
            #plt.axvspan(a,b,color='y',alpha=0.5,lw=1.0)
            if maxv[2]<b-a:
                maxv=(a,b,b-a)
    plt.axvspan(maxv[0],maxv[1],color='g',alpha=0.7,lw=0)
    plt.plot(df.Episode.as_matrix(),df.Steps.as_matrix())
    plt.xlim((0,200))
