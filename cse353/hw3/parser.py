f=open(raw_input("What is the name of the input file? It should be a csv without headers and the first column is the labels.\n"),"r")
blah=open(raw_input("What should the output be named?\n"),"w")

data = f.read().split("\n")

data.pop()
for datum in data:
    datum=datum.split(",")
    blah.write(datum[0])
    i=1
    while i<len(datum):
        blah.write(" "+str(i)+":"+datum[i])
        i=i+1
    blah.write("\n")
