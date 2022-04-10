import numpy as np
input_matrix=[[47,32,1,0,0,0,-2115,-1280,-45],
   [0,0,0,47,32,1,-1880,-1440,-40],
  [100,90,1,0,0,0,-9800,-7560,-98],
   [0,0,0,100,90,1,-8400,-8820,-84],
  [162,144,1,0,0,0,-25758,-20016,-159],
  [0,0,0,162,144,1,-22518,-22896,-139],
  [227,218,1,0,0,0,-51529,-44690,-227],
   [0,0,0,227,218,1,-46535,-49486,-205],
  [300,283,1,0,0,0,-91200,-77259,-304],
   [0,0,0,300,283,1,-81900,-86032,-273]]
input_matrix=np.array(input_matrix)

# computing multiplication of A-transpose and A to find the smallest value in each row

Homo=[]
def scalar(input_matrix):
    Result_matrix= np.dot(input_matrix.T,input_matrix)
    for i in Result_matrix:
        Homo.append(min(i))
    return Homo
print(scalar(input_matrix))
