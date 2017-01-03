import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plotdata(ax,value,base,basename,experiements):
    """Function that plot features against base value.
    Args:
        * ax: the subplot (plt.subplot).
        * value: The feature or parameter name ('advantage' , 'optimizer' etc...)
        * base: column of data of the same size of value
        * basename: the label for x axis
        * experiements: the main data set (pandas dataframe).
    return:
        * ax as a scatter plot."""
    
    tmp1 = experiements[value].unique()
    tmp2 = experiements[value].unique()
    if isinstance(tmp1[0],str):
        le = LabelEncoder()
        le.fit(tmp1)
        transformed =le.transform(experiements[value]) 
        ax.scatter( le.transform(experiements[value]),base)
        for i in range(experiements.shape[0]):
            ax.annotate(experiements.iloc[i]['experiment'],(transformed[i],base[i]),color='red')
        tmp1 = le.transform(tmp1)
    else:
        ax.scatter(experiements[value],base)
        for i in range(experiements.shape[0]):
            ax.annotate(experiements.iloc[i]['experiment'],(experiements.iloc[i][value],base[i]),color='red')
    ax.set_xlabel(value)
    ax.set_ylabel(basename)
    ax.set_title(value.upper())
    ax.xlim= (min(tmp1),max(tmp1))
    ax.set_xticks(tmp1)
    ax.set_xticklabels(tmp2,rotation=70)

    return ax

def plotsteps(eid,x,y,z,strng,vanish):
    font = FontProperties()
    font.set_weight('bold')
    font.set_size(15)
    exp = pd.read_csv('{}/exp_details.csv'.format(eid),header=None)
    test = exp[exp[7]=='Test']
    train= exp[exp[7]=='train']
    ax = plt.subplot(x,y,z+1)
    ax.text(0,1200,strng,color='green',fontproperties=font)
    ax.set_title('Training model')
    ax.plot(train[0],train[1])
    #ax.axvspan(train[0]*0.75,train[0].max(),color='red',alpha=0.5)
    splitpoint = train[0].max()*vanish
    ax.axvspan(0,splitpoint, color='green', alpha=0.8)
    ax.axvspan(splitpoint,train[0].max() , color='yellow', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax = plt.subplot(x,y,z+2)
    ax.plot(test[0],test[1])
    ax.axvspan(0,splitpoint/10 , color='green', alpha=0.8)
    ax.axvspan(splitpoint/10,test[0].max() , color='yellow', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Testing model')

def plotreward(eid,x,y,z,strng,vanish):
    font = FontProperties()
    font.set_weight('bold')
    font.set_size(15)
    exp = pd.read_csv('{}/exp_details.csv'.format(eid),header=None)
    test = exp[exp[7]=='Test']
    train= exp[exp[7]=='train']
    ax = plt.subplot(x,y,z+1)
    ax.text(0,1200,strng,color='green',fontproperties=font)
    ax.set_title('Training model')
    ax.plot(train[0],train[2])
    splitpoint = train[0].max()*vanish
    ax.axvspan(0,splitpoint, color='green', alpha=0.8)
    ax.axvspan(splitpoint,train[0].max() , color='yellow', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax = plt.subplot(x,y,z+2)
    ax.plot(test[0],test[2])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.axvspan(0,splitpoint/10 , color='green', alpha=0.8)
    ax.axvspan(splitpoint/10,test[0].max() , color='yellow', alpha=0.8)
    ax.set_title('Testing model')
