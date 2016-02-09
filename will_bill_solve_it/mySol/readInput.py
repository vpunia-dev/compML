import csv

def readHeader(inputFile):
    input = list(csv.reader(open(inputFile,'r')))
    header = input[0]
    headerDict = dict(zip(header,range(len(header))))
    # print header
    return headerDict

def readData(inputFile, featureList):
    headerDict = readHeader(inputFile)
    input = list(csv.reader(open(inputFile,'r')))
    data = input[1:]
    featureIndex = []
    for elem in featureList:
        try:
            featureIndex.append(headerDict[elem])
        except KeyError:
            print("key "+elem+" is not the right feature. Please verify the input file.")

    outData = [[elem[i] for i in featureIndex] for elem in data]
    return outData
