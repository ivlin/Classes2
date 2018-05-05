import numpy as np

def csv_as_matrix(filename):
    datafile = open(filename,'r')
    data=[]
    for row in datafile.read().split('\n'):
        try:
            data.append([float(col) for col in row.split(',')])
        except ValueError:
            print "INVALID LINE: line skipped"
    datafile.close()
    return data

def dot_product(a,b):
    if len(a)==len(b):
        return np.dot(a,b)
    return None

def init_weights(size):
    return np.zeros(size)

def perceptron_learning(dataset, update_modifier):
    weights=init_weights(len(dataset[0])-1)
    iteration=0

    current_min=None#minimum
    previous_drop=None#iteration number of the last decrease

    previous_err=None
    while True:
        print "iteration " +str(iteration)+" "+str(update_modifier)
        positive=True
        error=init_weights(len(weights))
        sum_err=0
        for data_point in dataset:
            inner_prod = dot_product(weights, data_point[1:])
            if data_point[0]*inner_prod <= 0:
                positive = False
                error = error + data_point[0]*data_point[1:]
                sum_err = sum_err+1
        weights = weights + update_modifier/len(error)*error
        if positive:
            return weights
        if previous_err is not None and sum_err>previous_err:
            if update_modifier>0.00001:
                update_modifier=update_modifier*0.1
        if sum_err<current_min or current_min is None:
            current_min=sum_err
            previous_drop=iteration
        if iteration-previous_drop>5:
            return weights
        previous_err=sum_err
        print sum_err
        print "previous drop "+str(previous_drop)
        print "current min "+str(current_min)
        print ""
        iteration=iteration+1


def combinations(dataset):
    transformed_data = []
    for datum in dataset:
        td=datum[:4]
        a=4
        while a < len(datum):
            b=4
            while b < len(datum):
                if a!=b:
                    if datum[a]+datum[b]>=2:
                        td.append(1)
                    else:
                        td.append(0)
                b=b+1
            a=a+1
        transformed_data.append(td)
    return transformed_data

def clean(dataset):
    #normalize to under 1.0
    ind=1
    while ind < 5:
        max=0
        for datum in dataset:
            if datum[ind]>max:
                max=datum[ind]
        for datum in dataset:
            if max is not 0:
                datum[ind]=datum[1+ind]/max
        ind=ind+1
    while ind<len(dataset[0]):
        present=False
        for datum in dataset:
            if datum[ind]!=0:
                present=True
        if not present:
            print str(ind)+" unnecessary"
            for datum in data:
                datum.pop(ind)
        ind=ind+1
    return dataset

#Index 0: Team won the game (1 or -1)
#Index 1: Cluster ID (related to location)
#Index 2: Game mode*
#Index 3: Game type (e.g. "ranked")
#Index 4-End: Heroes (5 1 and 5 -1)
data = csv_as_matrix(raw_input("What is the name of the training data file?\n"))
print "Data loaded"
data=combinations(data)
print "New features added"
data=clean(data)
print "Removed unnecessary values"
for i in xrange(len(data)):
    data[i]=np.array(data[i])
print "Converted"
weights=perceptron_learning(data,0.01)
print weights