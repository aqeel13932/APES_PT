import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plotdata(ax,ax2,value,base,basename,experiements):
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
        ax2.scatter( le.transform(experiements[value]),base)
        for i in range(experiements.shape[0]):
            ax.annotate(experiements.iloc[i]['experiment'],(transformed[i],base[i]),color='red')
        tmp1 = le.transform(tmp1)
    else:
        ax.scatter(experiements[value],base)
        ax2.scatter(experiements[value],base)
        for i in range(experiements.shape[0]):
            ax.annotate(experiements.iloc[i]['experiment'],(experiements.iloc[i][value],base[i]),color='red')
    ax.set_xlabel(value)
    ax.set_ylabel(basename)
    ax.set_title(value.upper())
    ax.xlim= (min(tmp1),max(tmp1))
    ax.set_xticks(tmp1)
    ax.set_xticklabels(tmp2,rotation=70)

    ax2.set_xlabel(value)
    ax2.set_ylabel(basename)
    ax2.set_title(value.upper())
    ax2.xlim= (min(tmp1),max(tmp1))
    ax2.set_xticks(tmp1)
    ax2.set_xticklabels(tmp2,rotation=70)

    return ax

def ReadCSV(eid,vanish):
    exp = pd.read_csv('{}/exp_details.csv'.format(eid),header=None)
    test = exp[exp[7]=='Test']
    train= exp[exp[7]=='train']
    train = train.sort_values([0])
    test= test.sort_values([0])
    train['cum_sum']=train[1].cumsum()
    splitpoint = train[1].sum()*vanish
    splitpoint=splitpoint[0]
    splitpoint = train[train.cum_sum>=splitpoint].iloc[0][0]
    return train,test,splitpoint

def plotsteps(eid,x,y,z,strng,vanish):
    font = FontProperties()
    font.set_weight('bold')
    font.set_size(15)
    train,test,splitpoint = ReadCSV(eid,vanish)
    ax = plt.subplot(x,y,z+1)
    ax.set_title('Training model')
    ax.plot(train[0],train[1])
    ax.text(0,train[1].max()+0.2*train[1].max(),strng,color='green',fontproperties=font)
    #ax.axvspan(train[0]*0.75,train[0].max(),color='red',alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.axvspan(0,splitpoint, color='green', alpha=0.8)
    ax.axvspan(splitpoint,train[0].max() , color='yellow', alpha=0.8)
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
    train,test,splitpoint = ReadCSV(eid,vanish)
    ax = plt.subplot(x,y,z+1)
    ax.text(0,test[2].max()+0.2*test[2].max(),strng,color='green',fontproperties=font)
    ax.set_title('Training model')
    ax.plot(train[0],train[2])
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

def Hyperparameters(ax,value,experiements):
    tmp1 = experiements[value].unique()
    tmp2 = experiements[value].unique()
    if isinstance(tmp1[0],str):
        le = LabelEncoder()
        le.fit(tmp1)
        experiements[value] = le.transform(experiements[value])
        tmp1 = le.transform(tmp1)
    ax.scatter(experiements[value],experiements.tr_AR)
    ax.set_xlabel(value)
    ax.set_ylabel('Average Steps')
    ax.set_title(value.upper())
    ax.xlim= (min(tmp1),max(tmp1))
    ax.set_xticks(tmp1)
    ax.set_xticklabels(tmp2,rotation=70)
    return ax
    
def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
