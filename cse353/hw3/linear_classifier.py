def csv_as_matrix(filename):
    datafile = open(filename,'r')
    data_plus=[]
    data_minus=[]
    for row in datafile.read().split('\n'):
        try:
            parsed_row = [float(col) for col in row.split(',')]
            if parsed_row[0]==1:
                data_plus.append(parsed_row)
            else:
                data_minus.append(parsed_row)
        except ValueError:
            print "INVALID LINE: line skipped"
    datafile.close()
    return data_plus, data_minus

def dot_product(a,b):
    if len(a)==len(b):
        sum=0.0
        for i in xrange(len(a)):
            sum=sum+a[i]*b[i]
        return sum
    return None

def init_weights(size):
    return [0 for i in xrange(size)]


def linear_regression(dataset):
    pass


#Index 0: Team won the game (1 or -1)
#Index 1: Cluster ID (related to location)
#Index 2: Game mode
#Index 3: Game type (e.g. "ranked")
#Index 4-End: Heroes (5 1 and 5 -1)
data_plus, data_minus = csv_as_matrix(raw_input("What is the name of the training data file?\n"))



print data_plus
print data_minus
