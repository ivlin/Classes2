import random

f = open("lin_sep.csv","w")

sample_weight=[-16.0, -12.2, 11.5, 2.1, -0.5, 17.2, 20.2, -9.9]

def dot_product(a,b):
    if len(a)==len(b):
        sum=0.0
        for i in xrange(len(a)):
            sum=sum+a[i]*b[i]
        return sum
    return None

for i in xrange(1000000):
    data = [0]
    for i in xrange(len(sample_weight)):
        data.append(random.uniform(-1200,1200))
    if dot_product(sample_weight,data[1:])>0:
        data[0]=1.0
    else:
        data[0]=-1.0
    for i in xrange(len(data)):
        if i<len(data)-1:
            f.write(str(data[i])+",")
        else:
            f.write(str(data[i])+"\n")

f.close()


