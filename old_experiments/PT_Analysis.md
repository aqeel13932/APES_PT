

```python
import pandas as pd
import numpy as np
df = pd.read_csv('FinalResults.csv',header=None)
df.columns = ['model','type','Env','Steps','Reward','Time','Dom_Moving']
pd.options.display.max_rows=1000
```


```python
best = pd.read_csv('Summary_Reward.csv')
accsteps=pd.read_csv('Summary_Steps.csv')
best[best.columns[3:]] = best[best.columns[3:]]>0
best['Total']=np.sum(best[best.columns[3:]].as_matrix(),axis=1)
```


```python
# conditions= (best.Dominant_Moving==False)&(best.Env1==True)&(best.Env2==False)&(best.Env3==True)&\
#             (best.Env4==True)&(best.Env5==False)&(best.Env7==False)&(best.Env8==True)&(best.Env9==True)&\
# (best.Env10==True)&(best.Env11==False)&(best.Env12==False)
conditions= (best.Env1==True)&(best.Env2==False)&(best.Env3==True)&\
            (best.Env4==True)&(best.Env5==False)&(best.Env7==False)&(best.Env11==False)&(best.Env12==False)
best[(conditions)]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>type</th>
      <th>Dominant_Moving</th>
      <th>Env1</th>
      <th>Env2</th>
      <th>Env3</th>
      <th>Env4</th>
      <th>Env5</th>
      <th>Env6</th>
      <th>Env7</th>
      <th>Env8</th>
      <th>Env9</th>
      <th>Env10</th>
      <th>Env11</th>
      <th>Env12</th>
      <th>Env13</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>647</td>
      <td>train</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>647</td>
      <td>target</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>59</th>
      <td>673</td>
      <td>target</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
accsteps[(conditions)]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>type</th>
      <th>Dominant_Moving</th>
      <th>Env1</th>
      <th>Env2</th>
      <th>Env3</th>
      <th>Env4</th>
      <th>Env5</th>
      <th>Env6</th>
      <th>Env7</th>
      <th>Env8</th>
      <th>Env9</th>
      <th>Env10</th>
      <th>Env11</th>
      <th>Env12</th>
      <th>Env13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>647</td>
      <td>train</td>
      <td>False</td>
      <td>5</td>
      <td>999</td>
      <td>5</td>
      <td>5</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>13</td>
      <td>999</td>
      <td>999</td>
      <td>-100.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>647</td>
      <td>target</td>
      <td>False</td>
      <td>5</td>
      <td>999</td>
      <td>5</td>
      <td>26</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>26</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>-100.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>673</td>
      <td>target</td>
      <td>False</td>
      <td>3</td>
      <td>999</td>
      <td>3</td>
      <td>3</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>3</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>999</td>
      <td>-100.0</td>
    </tr>
  </tbody>
</table>
</div>



<img src="./../pt_envs.png"></img>


```python

```

<img src="./../envs.png"></img>

# Backup code

## This code launched only once per adding new simulations
accomplished=pd.DataFrame(columns=['model','type','Dominant_Moving','Env1','Env2','Env3','Env4','Env5','Env6','Env7','Env8','Env9','Env10','Env11','Env12','Env13'])
accsteps = pd.DataFrame(columns=['model','type','Dominant_Moving','Env1','Env2','Env3','Env4','Env5','Env6','Env7','Env8','Env9','Env10','Env11','Env12','Env13'])
for i in df.model.unique():
    print(i,)
    models=['train','target']
    for mod in models:
        domagnt=[False]
        for dom in domagnt:
            curr= df[(df.model==i)&(df.Dom_Moving==dom)&(df.type==mod)]
            accomplished=accomplished.append(pd.DataFrame([[i,mod,dom,
                                       curr[curr.Env==1].Reward.as_matrix()[0],
                                      curr[curr.Env==2].Reward.as_matrix()[0],
                                      curr[curr.Env==3].Reward.as_matrix()[0],
                                      curr[curr.Env==4].Reward.as_matrix()[0],
                                      curr[curr.Env==5].Reward.as_matrix()[0],
                                      curr[curr.Env==6].Reward.as_matrix()[0],
                                      curr[curr.Env==7].Reward.as_matrix()[0],
                                      curr[curr.Env==8].Reward.as_matrix()[0],
                                      curr[curr.Env==9].Reward.as_matrix()[0],
                                      curr[curr.Env==10].Reward.as_matrix()[0],
                                      curr[curr.Env==11].Reward.as_matrix()[0],
                                      curr[curr.Env==12].Reward.as_matrix()[0],
                                      curr[curr.Env==13].Reward.as_matrix()[0]]],columns=accomplished.columns))
            accsteps=accsteps.append(pd.DataFrame([[i,mod,dom,
                                       curr[curr.Env==1].Steps.as_matrix()[0],
                                      curr[curr.Env==2].Steps.as_matrix()[0],
                                      curr[curr.Env==3].Steps.as_matrix()[0],
                                      curr[curr.Env==4].Steps.as_matrix()[0],
                                      curr[curr.Env==5].Steps.as_matrix()[0],
                                      curr[curr.Env==6].Steps.as_matrix()[0],
                                      curr[curr.Env==7].Steps.as_matrix()[0],
                                      curr[curr.Env==8].Steps.as_matrix()[0],
                                      curr[curr.Env==9].Steps.as_matrix()[0],
                                      curr[curr.Env==10].Steps.as_matrix()[0],
                                      curr[curr.Env==11].Steps.as_matrix()[0],
                                      curr[curr.Env==12].Steps.as_matrix()[0],
                                      curr[curr.Env==13].Reward.as_matrix()[0]]],columns=accsteps.columns))
accomplished.to_csv('Summary_Reward.csv',index=False)
accsteps.to_csv('Summary_Steps.csv',index=False)


```python
wmod ={}
for i in df.model.unique():
    models=['train','target']
    for mod in models:
        domagnt=[True,False]
        for dom in domagnt:
            curr= df[(df.model==i)&(df.Dom_Moving==dom)&(df.type==mod)]
            
            curr = curr[((curr.Env==1)&(curr.Reward>0))|
                        ((curr.Env==2)&(curr.Reward<0))|
                        ((curr.Env==3)&(curr.Reward<0))|
                        ((curr.Env==4)&(curr.Reward>0))|
                        #((curr.Env==5)&(curr.Reward<0))|
                        #((curr.Env==6)&(curr.Reward<0))|
                        #((curr.Env==7)&(curr.Reward<0))|
                        ((curr.Env==8)&(curr.Reward>0))|
                        ((curr.Env==9)&(curr.Reward<0))|
                        ((curr.Env==10)&(curr.Reward>0))]#|
                        #((curr.Env==11)&(curr.Reward>0))|
                        #((curr.Env==12)&(curr.Reward>0))]
            wmod[i]=curr
```


```python
accsteps[accsteps.model==406]
```


```python
import io
import base64
from IPython.display import HTML

video = io.open('/home/aqeel/Videos/other/Rurouni Kenshin-81.mkv', 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/avi;base64,{0}" type="video/avi" />
             </video>'''.format(encoded.decode('ascii')))
```

<video alt="test" controls>
                <source src="data:Rurouni Kenshin - 81.mkv;base64,{0}" type="video/avi" />
             </video>


```python
accsteps[(best.Env11==True)].sort_values('Env11')
```


```python
def GetModelName(model,Env):
    best[best.model==187][best.columns[:3]]
```
