import numpy as np
#Same direction
ego_map = dict.fromkeys([('N','N'),('W','W'),('S','S'),('E','E')],np.array([1,0,0,0]))
#Oppesite direction
ego_map.update(dict.fromkeys([('N','S'),('S','N'),('E,W'),('W','E')],np.array([0,1,0,0])))
#Looking left
ego_map.update(dict.fromkeys([('N','W'),('S','E'),('E','N'),('W','S')],np.array([0,0,1,0])))
#Looking right
ego_map.update(dict.fromkeys([('N','E'),('S','W'),('E','S'),('W','N')],np.array([0,0,0,1])))
def Ego_centric_ori(my,other):
    if other:
        return ego_map[(my,other)]
    else:
        result = np.zeros(4)
    return result

print(Ego_centric_ori('N','N'))
print(Ego_centric_ori('N','S'))
print(Ego_centric_ori('N','W'))
print(Ego_centric_ori('N','E'))
print(Ego_centric_ori('E','N'))
print(Ego_centric_ori('S','N'))
print(Ego_centric_ori('S',None))
