import numpy as np
import random

def dot(r,t):
    sum=0.0
    for i in xrange(len(r)):
        sum=sum+r[i]+t[i]
    return sum

x=[]
y=[]

for i in xrange(100000000):
    x.append(random.randint(0,1000))
    y.append(random.randint(0,1000))

dot(x,y)