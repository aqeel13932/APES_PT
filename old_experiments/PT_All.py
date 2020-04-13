import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',default=[],type=int)
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--cnn',type=bool,default=False)
args = parser.parse_args()
if args.cnn:
    print('using CNN Model')
else:
    print('Non CNN Model')
msgs=[',should go',',should not go']
msg='Env:{}'
'''
#Specific Cases
preferences={
    1:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    2:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(6,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    3:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(5,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    4:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(4,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]},
    5:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    6:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(6,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    7:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(5,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
    8:{'sub':(2,1),'dom':(2,5),'food':(3,4),'obs':(4,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]}

}
'''
'''
preferences={
1:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
2:{'sub':(2,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
3:{'sub':(2,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]},
4:{'sub':(2,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
5:{'sub':(2,1),'dom':(2,9),'food':(3,6),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' explore'},
6:{'sub':(2,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+' analyze the results'},
7:{'sub':(2,1),'dom':(4,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+' surprise me'},
8:{'sub':(2,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]},
9:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]},
10:{'sub':(2,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]},
11:{'sub':(9,1),'dom':(7,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+' race'},
12:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+' SM'},
13:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+' SM'},
14:{'sub':(3,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+' SM'},
15:{'sub':(3,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+' SM'},
16:{'sub':(2,1),'dom':(2,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]}
        }
'''
'''
preferences={
    408:{'sub':(2,1),'dom':(9,7),'food':(3-1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'}
}
'''
'''
preferences={
#Group 1 (Dominant positions) Original
18:{'sub':(2,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
19:{'sub':(2,1),'dom':(8,6),'food':(4,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_N_eat'},


#Group 1-1 shifting all elements.
118:{'sub':(2+1,1),'dom':(10-1,5),'food':(3-1,4),'obs':(3-1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
119:{'sub':(2+1,1),'dom':(8+1,6),'food':(4+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_N_eat'},


#Group 1-2 shifting only dominant when is observable.
218:{'sub':(2,1),'dom':(10-1,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
219:{'sub':(2,1),'dom':(8+1,6),'food':(4,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_N_eat'},

#Group 1-3 shifting subordinate only.
318:{'sub':(2+1,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
319:{'sub':(2+1,1),'dom':(8,6),'food':(4,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_N_eat'},

#Group 1-4 shifting only food.
418:{'sub':(2,1),'dom':(10,5),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
419:{'sub':(2,1),'dom':(8,6),'food':(4+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_N_eat'},
}
'''
'''
#Group 1 (Dominant positions) Original
1:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
2:{'sub':(2,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
3:{'sub':(2,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
4:{'sub':(2,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
8:{'sub':(2,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 1-1 shifting all elements.
101:{'sub':(2+1,1),'dom':(2+1,9),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
102:{'sub':(2+1,1),'dom':(7+1,4),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
103:{'sub':(2-1,1),'dom':(10-1,0),'food':(3-1,4),'obs':(3+1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
104:{'sub':(2+1,1),'dom':(9+1,9),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
108:{'sub':(2+1,1),'dom':(9+1,7),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 1-2 shifting only dominant when is observable.
201:{'sub':(2,1),'dom':(2+1,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
202:{'sub':(2,1),'dom':(7+1,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
203:{'sub':(2,1),'dom':(10-1,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
204:{'sub':(2,1),'dom':(9+1,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
208:{'sub':(2,1),'dom':(9+1,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 1-3 shifting subordinate only.
301:{'sub':(2+1,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
302:{'sub':(2+1,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
303:{'sub':(2+1,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
304:{'sub':(2+1,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
308:{'sub':(2+1,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 1-4 shifting only food.
401:{'sub':(2,1),'dom':(2,9),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
402:{'sub':(2,1),'dom':(7,4),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
403:{'sub':(2,1),'dom':(10,0),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
404:{'sub':(2,1),'dom':(9,9),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
,
#Group 2 Food, Obstacle Positions.
5:{'sub':(2,1),'dom':(2,9),'food':(3,6),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
6:{'sub':(2,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
9:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
10:{'sub':(2,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
16:{'sub':(2,1),'dom':(2,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-1 shifting all elements.
105:{'sub':(2+1,1),'dom':(2+1,9),'food':(3+1,6),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
106:{'sub':(2+1,1),'dom':(2+1,9),'food':(6+1,5),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
109:{'sub':(2+1,1),'dom':(2+1,9),'food':(3+1,4),'obs':(7+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
110:{'sub':(2+1,1),'dom':(2+1,9),'food':(7+1,4),'obs':(7+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
116:{'sub':(2+1,1),'dom':(2+1,9),'food':(1+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-2 shifting only dominant when is observable.
205:{'sub':(2,1),'dom':(2+1,9),'food':(3,6),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
206:{'sub':(2,1),'dom':(2+1,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
209:{'sub':(2,1),'dom':(2+1,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
210:{'sub':(2,1),'dom':(2+1,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
216:{'sub':(2,1),'dom':(2+1,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-3 shifting subordinate only.
305:{'sub':(2+1,1),'dom':(2,9),'food':(3,6),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
306:{'sub':(2+1,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
309:{'sub':(2+1,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
310:{'sub':(2+1,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
316:{'sub':(2+1,1),'dom':(2,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-4 shifting only food.
405:{'sub':(2,1),'dom':(2,9),'food':(3+1,6),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
406:{'sub':(2,1),'dom':(2,9),'food':(6+1,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
409:{'sub':(2,1),'dom':(2,9),'food':(3+1,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
410:{'sub':(2,1),'dom':(2,9),'food':(7+1,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
416:{'sub':(2,1),'dom':(2,9),'food':(1+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},   
#Group 3 not in Training Examples.
17:{'sub':(2,1),'dom':(4,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
11:{'sub':(9,1),'dom':(7,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
12:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
13:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
14:{'sub':(3,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
15:{'sub':(3,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-1 shifting all elements.
117:{'sub':(2+1,1),'dom':(4+1,1),'food':(6+1,5),'obs':(3+1,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
111:{'sub':(9+1,1),'dom':(7+1,1),'food':(8+1,10),'obs':(3+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
112:{'sub':(2+1,1),'dom':(2+1,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
113:{'sub':(2+1,1),'dom':(2+1,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
114:{'sub':(3+1,1),'dom':(3+1,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
115:{'sub':(3+1,1),'dom':(3+1,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-2 shifting only dominant when is observable.
217:{'sub':(2,1),'dom':(4+1,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
211:{'sub':(9,1),'dom':(7+1,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
212:{'sub':(2,1),'dom':(2+1,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
213:{'sub':(2,1),'dom':(2+1,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
214:{'sub':(3,1),'dom':(3+1,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
215:{'sub':(3,1),'dom':(3+1,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-3 shifting subordinate only.
317:{'sub':(2+1,1),'dom':(4,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
311:{'sub':(9+1,1),'dom':(7,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
312:{'sub':(2+1,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
313:{'sub':(2+1,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
314:{'sub':(3+1,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
315:{'sub':(3+1,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-4 shifting only food.
417:{'sub':(2,1),'dom':(4,1),'food':(6+1,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
411:{'sub':(9,1),'dom':(7,1),'food':(8+1,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
412:{'sub':(2,1),'dom':(2,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
413:{'sub':(2,1),'dom':(2,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
414:{'sub':(3,1),'dom':(3,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
415:{'sub':(3,1),'dom':(3,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'}
        }
        '''
#FINAL TEST CASES (USED IN THE PAPER).
'''
preferences={
#Group 1 (Dominant positions) Original
1:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
2:{'sub':(2,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
3:{'sub':(2,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
4:{'sub':(2,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
5:{'sub':(2,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
6:{'sub':(2,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},

#Group 1-1 shifting all elements.
101:{'sub':(2+1,1),'dom':(2+1,9),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
102:{'sub':(2+1,1),'dom':(7+1,4),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
103:{'sub':(2-1,1),'dom':(10-1,0),'food':(3-1,4),'obs':(3+1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
104:{'sub':(2+1,1),'dom':(9+1,9),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
105:{'sub':(2+1,1),'dom':(10-1,5),'food':(3-1,4),'obs':(3-1,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
106:{'sub':(2+1,1),'dom':(9+1,7),'food':(3+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 1-2 shifting only dominant when is observable.
201:{'sub':(2,1),'dom':(2+1,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
202:{'sub':(2,1),'dom':(7+1,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
203:{'sub':(2,1),'dom':(10-1,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
204:{'sub':(2,1),'dom':(9+1,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
205:{'sub':(2,1),'dom':(10-1,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
206:{'sub':(2,1),'dom':(9+1,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 1-3 shifting subordinate only.
301:{'sub':(2+1,1),'dom':(2,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
302:{'sub':(2+1,1),'dom':(7,4),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
303:{'sub':(2+1,1),'dom':(10,0),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
304:{'sub':(2+1,1),'dom':(9,9),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
305:{'sub':(2+1,1),'dom':(10,5),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
306:{'sub':(2+1,1),'dom':(9,7),'food':(3,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 1-4 shifting only food.
401:{'sub':(2,1),'dom':(2,9),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
402:{'sub':(2,1),'dom':(7,4),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[1]+'_avoid'},
403:{'sub':(2,1),'dom':(10,0),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
404:{'sub':(2,1),'dom':(9,9),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
405:{'sub':(2,1),'dom':(10,5),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'N','mesg':msg+msgs[0]+'_N_eat'},
406:{'sub':(2,1),'dom':(7,7),'food':(3+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
#Group 2 Food, Obstacle Positions.

7:{'sub':(2,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
8:{'sub':(2,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
9:{'sub':(2,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
10:{'sub':(2,1),'dom':(2,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-1 shifting all elements.

107:{'sub':(2+1,1),'dom':(2+1,9),'food':(6+1,5),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
108:{'sub':(2+1,1),'dom':(2+1,9),'food':(3+1,4),'obs':(7+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
109:{'sub':(2+1,1),'dom':(2+1,9),'food':(7+1,4),'obs':(7+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
110:{'sub':(2+1,1),'dom':(2+1,9),'food':(1+1,4),'obs':(3+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-2 shifting only dominant when is observable.

207:{'sub':(2,1),'dom':(2+1,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
208:{'sub':(2,1),'dom':(2+1,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
209:{'sub':(2,1),'dom':(2+1,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
210:{'sub':(2,1),'dom':(2+1,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-3 shifting subordinate only.

307:{'sub':(2+1,1),'dom':(2,9),'food':(6,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
308:{'sub':(2+1,1),'dom':(2,9),'food':(3,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
309:{'sub':(2+1,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
310:{'sub':(2+1,1),'dom':(2,9),'food':(1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-4 shifting only food.

407:{'sub':(2,1),'dom':(2,9),'food':(6+1,5),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_avoid'},
408:{'sub':(2,1),'dom':(2,9),'food':(3+1,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
409:{'sub':(2,1),'dom':(2,9),'food':(7+1,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
410:{'sub':(2,1),'dom':(2,9),'food':(1+1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'}
}
'''
preferences={
11:{'sub':(5,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
12:{'sub':(5,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-1 shifting all elements.
111:{'sub':(5+1,1),'dom':(2+1,9),'food':(7+1,4),'obs':(7+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
112:{'sub':(5+1,1),'dom':(2+1,9),'food':(7+1,4),'obs':(7+1,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-2 shifting only dominant when is observable.
211:{'sub':(5,1),'dom':(2+1,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
212:{'sub':(5,1),'dom':(2+1,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-3 shifting subordinate only.
311:{'sub':(5+1,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
312:{'sub':(5+1,1),'dom':(2,9),'food':(7,4),'obs':(7,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
#Group 2-4 shifting only food.
411:{'sub':(5,1),'dom':(2,9),'food':(7+1,4),'obs':(7,5),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_eat'},
412:{'sub':(5,1),'dom':(2,9),'food':(7+1,4),'obs':(7,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
}
import numpy as np
np.random.seed(4917)
from keras.models import load_model
import skvideo.io
from time import time
from Settings import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from Environments import CreateEnvironment

def AddTextToImage(img,action,AgentView=0):
    img = np.array(img*255,dtype=np.uint8)
    img = Image.fromarray(img)
    #img = Image.fromarray(game.BuildImage())
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("LiberationSans-Bold.ttf", 12)
    # draw.text((x, y),"Sample Text",(r,g,b))
    if AgentView:
        draw.text((0, 0),"Action:{}".format(action),(255,0,0),font=font)
    else:
        draw.text((0, 0),"Action:{}".format(action),(0,0,0),font=font)
    return img


def WriteInfo(model_id,mod_type,env,steps,reward,time,dom):#,rwsc,rwprob,aiproba,eptype,trqavg,tsqavg):
    with open('Final_Results/FinalResults.csv','a') as outp:
        print('{},{},{},{},{},{},{}'.format(model_id,mod_type,env,steps,reward,time,dom))
        outp.write('{},{},{},{},{},{},{}\n'.format(model_id,mod_type,env,steps,reward,time,dom))
#### Load the model
for i in args.models:
    if not os.path.exists('Final_Results/{}'.format(i)):
        os.makedirs('Final_Results/{}'.format(i))
    train = load_model('output/{}/MOD/model.h5'.format(i))
    #target = load_model('output/{}/MOD/target_model.h5'.format(i))
    models={'train':train}#,'target':target}
    #models={'target':target}
    for mod in models:
        np.random.seed(1337)
        model = models[mod]
        #domagnt=[True,False]
        domagnt=[False]
        for dom in domagnt:
            #for env in range(1,13):
            for env in preferences:
                counter = env+(env-1)*2
                preferences[env]['mesg']=preferences[env]['mesg'].format(env)
                game = CreateEnvironment(preferences[env],args.naction)
                DAgent,AIAgent = [game.agents[x] for x in game.agents]
                AIAgent = game.agents[1001]
                #print('D:{},AI:{}'.format(DAgent.ID,AIAgent.ID))
                #AIAgent = [game.agents[x] for x in game.agents][0]
                TestingCounter=0
                TestingCounter+=1
                writer = skvideo.io.FFmpegWriter("Final_Results/{}/ENV:{},mod:{},dom:{}.avi".format(i,env,mod,dom))
                writer2 = skvideo.io.FFmpegWriter("Final_Results/{}/ENV:{},mod:{},dom:{}_AG.avi".format(i,env,mod,dom))
                #game.GenerateWorld()
                img = game.BuildImage()
                game.Step()
                #plt.imsave('Final_Results/{}/ENV_{}.png'.format(i,env,TestingCounter),img)
                Start = time.time()
                episode_reward=0
                writer.writeFrame(AddTextToImage(game.BuildImage(),AIAgent.NextAction,0))
                writer2.writeFrame(AddTextToImage(game.AgentViewPoint(AIAgent.ID),AIAgent.NextAction,1))
                for t in range(1000):
                    if args.cnn:
                        cnn,rest = AIAgent.Convlutional_output()
                        q = model.predict([cnn[None,:],rest[None,:]], batch_size=1)
                    else:
                        #AIAgent.NNFeed['agentori1002'].fill(0)
                        observation = AIAgent.Flateoutput()
                        s =np.array([observation])
                        q = model.predict(s, batch_size=1)
                    #if t==1:
                    #print('Input type:',AIAgent.Flateoutput())
                    #print('Q value type:',q)
                    #if np.random.random()<0.05:
                    #    action=AIAgent.RandomAction()
                    #else:
                        #action = np.argmax(q[0,:-1])
                    action = np.argmax(q[0])
                    #print(Settings.PossibleActions[action],action)
                    AIAgent.NextAction = Settings.PossibleActions[action]
                    #print(AIAgent.NextAction)
                    #if env not in [4,5]:
                    if dom:
                        DAgent.DetectAndAstar()
                    #print(DAgent.NextAction)
                    game.Step()
                    if args.cnn:
                        cnn,rest = AIAgent.Convlutional_output()
                    else:
                        observation = AIAgent.Flateoutput()
                    reward = AIAgent.CurrentReward
                    #print(reward)
                    done = game.Terminated[0]
                    #observation, reward, done, info = env.step(action)
                    episode_reward += reward
                    writer.writeFrame(AddTextToImage(game.BuildImage(),'{},TR:{}'.format(AIAgent.NextAction,episode_reward),0))
                    writer2.writeFrame(AddTextToImage(game.AgentViewPoint(AIAgent.ID),'{},TR:{}'.format(AIAgent.NextAction,episode_reward),1))
                    if done:
                        break

                writer.close()
                writer2.close()
                Start = time.time()-Start
                WriteInfo(i,mod,env,t,episode_reward,Start,dom)

