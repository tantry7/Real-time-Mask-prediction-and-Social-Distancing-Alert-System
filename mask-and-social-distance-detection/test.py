import numpy
import numpy as np

l = [23,4,5,6,324,6,8,6,7,877]
s = np.arange(0,50)
def list_l(t):
    print(t[0],t[1])
    print(t[len(t)//2-1],t[len(t)//2])
    print(t[len(t)-1],t[len(t)-2])

list_l(l)
list_l(s)
