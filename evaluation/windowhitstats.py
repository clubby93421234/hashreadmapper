import sys
import statistics

if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], "windowhitstats.txt truehitsperwindow.txt")
    sys.exit(0)

#both input files must have 1 line per window, sorted by chromosomeId, windowId

#windowhitstats.txt
file = open(sys.argv[1])

#truehitsperwindow.txt
truehitsfile = open(sys.argv[2])


truehitnumbers = []
correcthitnumbers = []
incorrecthitnumbers = []

foo1 = [0,0,0,0,0,0,0]
foo2 = [0,0,0,0,0,0,0]

for line in truehitsfile:
    tokens = line.split(" ")
    if len(tokens) != 3:
        print(tokens)
    assert len(tokens) == 3
    truehitnumbers.append(int(tokens[2]))
    foo1[int(tokens[0])] += 1

for line in file:
    tokens = line.split(" ")
    assert len(tokens) == 4
    correcthitnumbers.append(int(tokens[2]))
    incorrecthitnumbers.append(int(tokens[3]))
    foo2[int(tokens[0])] += 1

if foo1 != foo2:
    print("error foo1 foo2")
    print(foo1)
    print(foo2)
    sys.exit(0)


rates = []
nohits = 0

truehitsratios = []

for index, (t,c,i) in enumerate(zip(truehitnumbers, correcthitnumbers, incorrecthitnumbers)):
    if t < c:
        print("error",index, t,c,i)
    if c+i == 0:
        nohits += 1        
    else:
        rates.append(float(c) / (float(c) + float(i)))
        if t == 0:
            truehitsratios.append(float(1))
        else:
            truehitsratios.append(float(c) / float(t))
    
numwindows = len(rates) + nohits
minrate = min(rates)
maxrate = max(rates)
avgrate = sum(rates) / len(rates)
medianrate = statistics.median(rates)

mintruerate = min(truehitsratios)
maxtruerate = max(truehitsratios)
avgtruerate = sum(truehitsratios) / len(truehitsratios)
mediantruerate = statistics.median(truehitsratios)

#print("numwindows: ", numwindows, ", windows without hits: ", nohits)
#print("minrate: ", round(minrate,4), ", maxrate: ", round(maxrate,4), ", avgrate: ", round(avgrate,4), ", medianrate: ", round(medianrate,4))
#print("mintruerate: ", round(mintruerate,4), ", maxtruerate: ", round(maxtruerate,4), ", avgtruerate: ", round(avgtruerate,4), \
    #", mediantruerate: ", round(mediantruerate,4))

print(round(avgrate,4),round(medianrate,4),round(avgtruerate,4),round(mediantruerate,4))




