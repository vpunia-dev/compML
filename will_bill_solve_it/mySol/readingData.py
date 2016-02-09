import readInput
import csv

class data:
    users = []
    problems = []
    submissions = []

    def __init__(self,users,problems,submissions):
        self.users = users
        self.problems = problems
        self.submissions = submissions
    # def process_submissions():


class testFields:
    users = []
    problem = []
    submissions = []

    def __init__(self):
        self.users = ['user_id','solved_count','attempts']
        # self.problem = ['problem_id','level','accuracy','solved_count','error_count','tag1','tag2','tag3','tag4','tag5']
        self.problem = ['problem_id','level','accuracy','solved_count','error_count','rating','tag1','tag2','tag3','tag4','tag5','level']
        self.test = ['user_id','problem_id']

class fields:
    users = []
    problem = []
    submissions = []

    def __init__(self):
        self.users = ['user_id','solved_count','attempts']
        # self.problem = ['problem_id','level','accuracy','solved_count','error_count','tag1','tag2','tag3','tag4','tag5']
        self.problem = ['problem_id','accuracy','solved_count','error_count','rating','tag1','tag2','tag3','tag4','tag5','level']
        self.submissions = ['user_id','problem_id','solved_status','result']


def newDataFile(filename,header,data):
        userF = csv.writer(open(filename,'w'))
        # userF.writerow(header)
        userF.writerows(data)


def printTags(data,offset):

    tags=[]
    for item in data:
        tags.extend(item[offset:])

    tags = set(tags)
    tags.remove('')
    return list(tags)

def processUsers(data):
    newData = []
    for item in data:
        temp = float(item[1])/(float(item[1])+float(item[2])+1)
        tempdata = [item[0]]
        tempdata.append(item[2])
        tempdata.append(temp)
        newData.append(tempdata)
    return newData

def processSubmissions(data):
    newData = []
    subDict  = dict()
    for item in data:
        if (item[0],item[1]) in subDict:
            subDict[(item[0],item[1])].append(item[-2:])
        else:
            subDict[(item[0],item[1])] =[]
            subDict[(item[0],item[1])].append(item[-2:])


    newData = []
    for item in subDict.keys():
        if ['SO','AC'] in subDict[item]:
            tempList = list(item)
            tempList.append(1)
            newData.append(tempList)
        elif ['SO','PAC'] in subDict[item]:
            tempList = list(item)
            tempList.append(0.5)
            newData.append(tempList)
        else:
            tempList = list(item)
            tempList.append(0)
            newData.append(tempList)

    return newData

def createInput(subs,users,probs):
    # userIds = [item[0] for item in users]
    # probIds = [item[0] for item in probs]
    #
    # newData = []
    #
    # print len(subs)
    # i=0
    # for item in subs:
    #     print i
    #     temp = []
    #     temp = users[userIds.index(item[0])][1:]
    #     temp.extend(probs[probIds.index(item[1])][1:])
    #     temp.append(item[-1])
    #     newData.append(temp)
    #     i = i+1
    userDict = dict()
    probDict = dict()

    for item in users:
        userDict[item[0]]=item[1:]
    for item in probs:
        probDict[item[0]]=item[1:]

    newData = []
    i=0
    for item in subs:
        if i%50000==0:
            print i
        temp = []
        temp.extend(userDict[item[0]])
        temp.extend(probDict[item[1]])
        temp.append(item[-1])
        newData.append(temp)
        i = i+1

    return newData

# def processProbs(data):
#     levels = ['V-E','E','E-M','M','M-H','H']
#     levelDict = dict(zip(levels,[2*i for i in range(1,len(levels)+1) ]))
#     newData = []
#     for item in data:
#         if item[1] in levelDict:
#             item[1]=levelDict[item[1]]
#             newData.append(item)
#         else:
#             item[1]=0
#             newData.append(item)
#
#     return newData

def manipFeatures(data):
    import finalManip
    temp = []
    # print data
    for feature in [finalManip.Basic,finalManip.Math,finalManip.DS,
                    finalManip.ADS,finalManip.Algo,finalManip.AAlgo,finalManip.Quality,finalManip.Random]:
        weight = 0

        for item in data:
            if item in feature:
                weight = weight+ feature[item]

        temp.append(weight)

    return temp

def processProbs(data):

    # ['problem_id','accuracy','solved_count','error_count','rating',featureWeight]
    tagList = ['','V-E','E','E-M','M','M-H','H']
    tagDict = dict(zip(tagList,range(len(tagList))))

    newData = []
    for item in data:
        temp = [item[0]]
        temp.append(item[1])
        temp.append(float(item[2])/( float(item[2]) + float(item[3])+1))
        temp.append(item[2])
        temp.append(item[4])

        if item[-1] in tagDict:
            temp.append(tagDict[item[-1]])
        else:
            temp.append(0)
        # print item
        temp.extend(manipFeatures(item[-6:-1]))
        newData.append(temp)
    return newData


def createTestInput(test,users,probs):

    userDict = dict()
    probDict = dict()

    for item in users:
        userDict[item[0]]=item[1:]
    for item in probs:
        probDict[item[0]]=item[1:]

    newData = []
    for item in test:
        temp = []
        temp.extend(userDict[item[0]])
        temp.extend(probDict[item[1]])
        newData.append(temp)
    return newData


if __name__== '__main__':

    import sys
    option = int(sys.argv[1])

    if option:
        fields = fields()
        data1 = data(readInput.readData('./train/users.csv',fields.users),
                    readInput.readData('./train/problems.csv',fields.problem),
                    readInput.readData('./train/submissions.csv',fields.submissions))


        #adding user accuracy
        data1.users = processUsers(data1.users)
        fields.users = ['user_id','attempts','usr_accuracy']

        data1.submissions =  processSubmissions(data1.submissions)
        fields.submissions = ['user_id','problem_id','solved_status']

        fields.problem = ['problem_id','accuracy','percent_solve','solve_count','rating','level','Basic','Math','DS',
                    'ADS','Algo','AAlgo','Quality','Random']
        data1.problems = processProbs(data1.problems)

        data = createInput(data1.submissions,data1.users,data1.problems)
        header = fields.users[1:]
        header.extend(fields.problem[1:])
        header.append('outcome')
        newDataFile('input.csv',header,data)

    else:

        fields = fields()
        testdata = readInput.readData('./test/test.csv',['user_id','problem_id'])
        data2 = data(readInput.readData('./test/users.csv',fields.users),
                    readInput.readData('./test/problems.csv',fields.problem),
                    testdata)
        #
        #
        #creates test input too
        data2.users = processUsers(data2.users)
        fields.users = ['user_id','attempts','usr_accuracy']
        #
        data2.problems = processProbs(data2.problems)
        fields.problem = ['problem_id','accuracy','percent_solve','solve_count','rating','level','Basic','Math','DS',
                    'ADS','Algo','AAlgo','Quality','Random']

        data = createTestInput(testdata,data2.users,data2.problems)
        header = fields.users[1:]
        header.extend(fields.problem[1:])

        data3 = createTestInput(data2.submissions,data2.users,data2.problems)
        header = fields.users[1:]
        header.extend(fields.problem[1:])
        newDataFile('testInput.csv',header,data3)



