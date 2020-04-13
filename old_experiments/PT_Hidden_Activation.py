import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',default=[],type=int)
parser.add_argument('--naction',type=int,default=0)
parser.add_argument('--cnn',type=bool,default=False)
args = parser.parse_args()

msgs=[',should go',',should not go']
msg='Env:{}'
'''
preferences={
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
408:{'sub':(2,1),'dom':(9,7),'food':(3-1,4),'obs':(3,5),'subdir':'E','domdir':'W','mesg':msg+msgs[0]+'_eat'},
    
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
7:{'sub':(2,1),'dom':(4,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
11:{'sub':(9,1),'dom':(7,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
12:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
13:{'sub':(2,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
14:{'sub':(3,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
15:{'sub':(3,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-1 shifting all elements.
107:{'sub':(2+1,1),'dom':(4+1,1),'food':(6+1,5),'obs':(3+1,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
111:{'sub':(9+1,1),'dom':(7+1,1),'food':(8+1,10),'obs':(3+1,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
112:{'sub':(2+1,1),'dom':(2+1,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
113:{'sub':(2+1,1),'dom':(2+1,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
114:{'sub':(3+1,1),'dom':(3+1,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
115:{'sub':(3+1,1),'dom':(3+1,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-2 shifting only dominant when is observable.
207:{'sub':(2,1),'dom':(4+1,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
211:{'sub':(9,1),'dom':(7+1,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
212:{'sub':(2,1),'dom':(2+1,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
213:{'sub':(2,1),'dom':(2+1,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
214:{'sub':(3,1),'dom':(3+1,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
215:{'sub':(3,1),'dom':(3+1,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-3 shifting subordinate only.
307:{'sub':(2+1,1),'dom':(4,1),'food':(6,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
311:{'sub':(9+1,1),'dom':(7,1),'food':(8,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
312:{'sub':(2+1,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
313:{'sub':(2+1,1),'dom':(2,9),'food':(1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
314:{'sub':(3+1,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
315:{'sub':(3+1,1),'dom':(3,9),'food':(2,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
#Group 3-4 shifting only food.
407:{'sub':(2,1),'dom':(4,1),'food':(6+1,5),'obs':(3,5),'subdir':'W','domdir':'W','mesg':msg+msgs[0]+'_N_avoid'},
411:{'sub':(9,1),'dom':(7,1),'food':(8+1,10),'obs':(3,5),'subdir':'E','domdir':'E','mesg':msg+msgs[1]+'_N_avoid'},
412:{'sub':(2,1),'dom':(2,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_eat'},
413:{'sub':(2,1),'dom':(2,9),'food':(1+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'},
414:{'sub':(3,1),'dom':(3,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'E','mesg':msg+msgs[0]+'_N_avoid'},
415:{'sub':(3,1),'dom':(3,9),'food':(2+1,5),'obs':(0,0),'subdir':'E','domdir':'W','mesg':msg+msgs[1]+'_N_avoid'}
        }
'''
#FINAL TEST CASES (USED IN THE PAPER).
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
import numpy as np
np.random.seed(4917)
from keras.models import load_model,Model
import skvideo.io
from time import time
from Settings import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from Environments import CreateEnvironment
counterrr=0
InputArray = np.zeros((len(preferences.keys()),634))
InputArray.fill(-1000)
HiddenActivation = np.zeros((len(preferences.keys()),101))
HiddenActivation.fill(-1000)

outputArray = np.zeros((len(preferences.keys()),7))
outputArray.fill(-1000)
#### Load the model
for i in args.models:
    if not os.path.exists('Final_Results/{}'.format(i)):
        os.makedirs('Final_Results/{}'.format(i))
    train = load_model('output/{}/MOD/model.h5'.format(i))
    #FM = Model(input=train.layers[0].input,output=train.layers[1].output)
    
    FM = Model(input=train.layers[0].input,output=train.layers[2].output)
    models={'train':FM}
    for mod in models:
        np.random.seed(1337)
        model = models[mod]
        model.summary()
        domagnt=[False]
        for dom in domagnt:
            for env in preferences:
                preferences[env]['mesg']=preferences[env]['mesg'].format(env)
                game = CreateEnvironment(preferences[env],args.naction)
                DAgent,AIAgent = [game.agents[x] for x in game.agents]
                img = game.BuildImage()
                game.Step()
                plt.imsave('Final_Results/{}/ENV_{}.png'.format(i,env),img)
                Start = time.time()
                episode_reward=0

                if args.cnn:
                    cnn,rest = AIAgent.Convlutional_output()
                    q = model.predict([cnn[None,:],rest[None,:]], batch_size=1)
                else:
                    observation = AIAgent.Flateoutput()
                    s =np.array([observation])
                    q = model.predict(s, batch_size=1)
                #InputArray[counterrr,1:]=s
                #InputArray[counterrr,0]=env
                #HiddenActivation[counterrr,1:]=q[0]
                #HiddenActivation[counterrr,0]=env
                outputArray[counterrr,1:]=q[0]
                outputArray[counterrr,0]=env
                counterrr+=1

                Start = time.time()-Start
#np.save('Input_Final_Cases.npy',InputArray)
#np.save('Hidden_Activation_Final_Cases.npy',HiddenActivation)
np.save('Output_Final_Cases.npy',outputArray)
print(counterrr)
