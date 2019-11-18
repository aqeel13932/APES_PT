import pandas as pd
for i in range(939,967):
    try:
        x = pd.read_csv('{}/exp_details.csv'.format(i),header=None)
        x.columns =['trail','steps','reward','qpre','4','5','6','type','time','9']
        train = x[x.type=='train']
        test = x[x.type=='Test']
        print('E: {} , tr_mean:{},ts_mean:{}'.format(i,train['reward'].mean(),test['reward'].mean()))
    except Exception as inst:
        print(inst)
        print(i)

