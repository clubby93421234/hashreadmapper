
import sys

if len(sys.argv) < 5:
    print("Usage:", sys.argv[0], "readwindows.txt genomeindex.fai windowsize kmersize")
    sys.exit(0)

nameToId = {
    "CHROMOSOME_I": 0,
    "CHROMOSOME_II": 1,
    "CHROMOSOME_III": 2,
    "CHROMOSOME_IV": 3,
    "CHROMOSOME_V": 4,
    "CHROMOSOME_X": 5,
    "CHROMOSOME_MtDNA": 6
}

#readwindows.txt 
file = open(sys.argv[1])

#c_elegans.WS222.genomic.fa.fai 
genomeindex = open(sys.argv[2])

windowsize = int(sys.argv[3])
k = int(sys.argv[4])

hitsPerWindow = {}

#init hitsPerWindow with count 0
for line in genomeindex:
    #line like: CHROMOSOME_I	15072423	14	50	51
    tokens = line.split("\t")
    name = tokens[0]
    assert name in nameToId
    chrid = nameToId[name]
    length = int(tokens[1])
    #sdiv(length, (windowsize - k + 1))
    numWindows = int((length + (windowsize - k + 1) - 1) / (windowsize - k + 1))
    
    for windowId in range(numWindows):
        hitsPerWindow[(chrid, windowId)] = 0


for line in file:
    tokens = line.split(" ")

    #chrname windowId
    assert(len(tokens) == 2)

    name = tokens[0]
    assert name in nameToId
    chrid = nameToId[name]

    for i in range(1, len(tokens)):
        windowId = int(tokens[i])
        tup = (chrid, windowId)
        assert tup in hitsPerWindow
        hitsPerWindow[tup] += 1



sortedhits = sorted(hitsPerWindow.items(), key=lambda x: x[0])

for ((chrId, windowId), count) in sortedhits:
    print(chrId, windowId, count)
