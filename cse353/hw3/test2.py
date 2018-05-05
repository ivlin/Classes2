import numpy as np
import random
x=[]
y=[]

for i in xrange(100000000):
    x.append(random.randint(0,1000))
    y.append(random.randint(0,1000))

x=np.array(x)
y=np.array(y)

np.dot(x,y)