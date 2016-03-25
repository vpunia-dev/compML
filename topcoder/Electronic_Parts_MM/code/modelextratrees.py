from datetime import datetime
import random
import numpy as np
from collections import Counter,defaultdict
thresholdyes = 0.05
thresholdno = -0.05
trainx = []
trainy = []

def l2reg(x):
	return np.dot(x,x)

def l1reg(x):
	return sum(x)

def errorfunc(w):
	totalerror = 0

	for i in range(len(trainx)):
		x = trainx[i]
		pred = np.dot(w,x[1:])

		if pred < -0.1:
			pred = -1
		elif pred > 0.1:
			pred = 1
		else:	
			pred = 0

		if trainy[i]==0 and pred==-1:
			totalerror += 0.2	

		if trainy[i]==1 and pred==-1:
			totalerror += 0.7

		if trainy[i]==-1 and pred==0:
			totalerror += 0.5

		if trainy[i]==1 and pred==0:
			totalerror += 0.01

		if trainy[i]==-1 and pred==1:
			totalerror += 1.0

		if trainy[i]==0 and pred==1:
			totalerror += 0.01

	return totalerror+l1reg(w)

def removeDuplicates(X,y):
	dictX = {}
	culprits = []
	for x,Y in zip(X,y):
	    if tuple(x) in dictX:
		if dictX[tuple(x)]!=Y:
			culprits.append(x)
	    else:
		dictX[tuple(x)] = Y

	print(len(X),len(culprits))
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

pTimes = {}
pPrices = {}
pCosts = {}
pFreq = {}
pTimeranges = {}

def productStats(data):
	global pTimes,pPrices,pCosts,pFreq,pTimeranges
	categorical = {}
	categorical[8] = ["A","B"]
	categorical[10] = ["1","2","3"]
	categorical[11] = ["A","B","C"]
	categorical[12] = ["N","L"]
	categorical[13] = ["ST","DM"]
	categorical[19] = ["IN_HOUSE","NOT_IN_HOUSE"]
	categorical[21] = ["Y","N"]
	categorical[25] = ["B","EA","LB"]
	categorical[26] = ["A","B"]
	categorical[27] = ["1","2","3","4","5"]

	datefields = [2,14]

	#dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	customerfields = [1,7,8,9,10,11,12,13,14]
	productfields = [5,6,19,21,25,26]
	constantfields = [12,10,11,13]
	#dropfields = []
	#dropfields = [2,14]
	productA = defaultdict(list)
	productB = defaultdict(list)
	for d in data:
		row = []
		for i in range(len(d)):
			if i in datefields:
				date = d[i].split(" ")[0]				
				year = int(date.split("-")[0])
				month = int(date.split("-")[1])
				day = int(date.split("-")[2])

				seconds = (datetime(year,month,day) - datetime(1970,1,1)).total_seconds()

				row.append(seconds)
			elif i not in categorical:
				if i!=28:
					row.append(float(d[i]))	
				else:
					if d[i]=="Yes":
						row.append(1)
					if d[i]=="No":
						row.append(-1)
					if d[i]=="Maybe":
						row.append(0)
			else:
				bitv = [0]*len(categorical[i])
				index=0
				for j in range(len(categorical[i])):
					if d[i]==categorical[i][j]:
						row.append(index)
						#bitv[j]=1
						break
					index+=1
				#row = row + bitv
		if d[8]=="A":
			productA[int(d[0])].append(row)
		else:
			productB[int(d[0])].append(row)

	featureA = []
	featureB = []
	for pid,feats in productA.items():
		f = np.array(feats)
		numtransactions = f.shape[0]
		times = f[:,2]
		times.sort()
		period = times[-1]-times[0]
		freq = 0.0
		for t1,t2 in zip(times,times[1:]):
		    freq += abs(t2-t1)
		freq = freq/(len(times)-1.0)

		prices = f[:,3]
		prices.sort()
		pricerange = prices[-1]-prices[0]
		avgprice = sum(prices)/len(prices)

		sales = f[:,4]
		totalsales = sum(sales)

		class1 = f[:,15][0]
		class2 = f[:,16][0]
		class3 = f[:,17][0]
		class4 = f[:,18][0]
		
		brand = f[:,19][0]
		pattr = f[:,20][0]
		totalweight = sum(f[:,22])
		totalboxes = sum(f[:,23])

		costs = f[:,24]
		costs.sort()
		costrange = costs[-1]-costs[0]
		avgcost = sum(costs)/len(costs)

		if f.shape[1]==29:
			labels = f[:,28]
			featureA.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost,labels[0]])
		else:
			featureA.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost])

	for pid,feats in productB.items():
		f = np.array(feats)
		numtransactions = f.shape[0]
		times = f[:,2]
		times.sort()
		period = times[-1]-times[0]
		freq = 0.0
		for t1,t2 in zip(times,times[1:]):
		    freq += abs(t2-t1)
		freq = freq/(len(times)-1.0)

		prices = f[:,3]
		prices.sort()
		pricerange = prices[-1]-prices[0]
		avgprice = sum(prices)/len(prices)

		sales = f[:,4]
		totalsales = sum(sales)

		class1 = f[:,15][0]
		class2 = f[:,16][0]
		class3 = f[:,17][0]
		class4 = f[:,18][0]
		
		brand = f[:,19][0]
		pattr = f[:,20][0]
		totalweight = sum(f[:,22])
		totalboxes = sum(f[:,23])

		costs = f[:,24]
		costs.sort()
		costrange = costs[-1]-costs[0]
		avgcost = sum(costs)/len(costs)

		if f.shape[1]==29:
			labels = f[:,28]
			featureB.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost,labels[0]])
		else:
			featureB.append([numtransactions,period,freq,pricerange,avgprice,totalsales,class1,class2,class3,class4,brand,pattr,totalweight,totalboxes,costrange,avgcost])

	return np.array(featureA),np.array(featureB)

def convertToFeatures(data):
	global pTimes,pPrices,pCosts,pFreq,pTimeranges
	categorical = {}
	categorical[8] = ["A","B"]
	categorical[10] = ["1","2","3"]
	categorical[11] = ["A","B","C"]
	categorical[12] = ["N","L"]
	categorical[13] = ["ST","DM"]
	categorical[19] = ["IN_HOUSE","NOT_IN_HOUSE"]
	categorical[21] = ["Y","N"]
	categorical[25] = ["B","EA","LB"]
	categorical[26] = ["A","B"]
	categorical[27] = ["1","2","3","4","5"]

	datefields = [2,14]

	#dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	#dropfields = [0,1,2,5,6,7,8,9,10,11,12,13,14,21,25,26,27]
	dropfields = [0,1,2,14]
	keepfields = [16,8,22,17,18,15,20,24,3,4]
	customerfields = [1,7,8,9,10,11,12,13,14]
	productfields = [5,6,19,21,25,26]
	constantfields = [12,10,11,13]
	#dropfields = []
	#dropfields = [2,14]
	featureA = []
	featureB = []
	for d in data:
		row = []
		for i in range(len(d)):
			if i not in keepfields:
				continue
			if i in datefields:
				date = d[i].split(" ")[0]				
				year = int(date.split("-")[0])
				month = int(date.split("-")[1])
				day = int(date.split("-")[2])

				seconds = (datetime(year,month,day) - datetime(1970,1,1)).total_seconds()

				#row.append(seconds)
				if int(d[0]) not in pTimes:
					pTimes[int(d[0])] = [seconds]
				else:
					pTimes[int(d[0])].append(seconds)
			elif i not in categorical:
				if i!=0:
					row.append(float(d[i]))	
				else:
					row.append(int(d[i]))
				if i==3:
					if int(d[0]) not in pPrices:
						pPrices[int(d[0])] = [float(d[i])]	
					else:
						pPrices[int(d[0])].append(float(d[i]))
				if i==24:
					if int(d[0]) not in pCosts:
						pCosts[int(d[0])] = [float(d[i])]	
					else:
						pCosts[int(d[0])].append(float(d[i]))
			else:
				bitv = [0]*len(categorical[i])
				index=0
				for j in range(len(categorical[i])):
					if d[i]==categorical[i][j]:
						row.append(index)
						#bitv[j]=1
						break
					index+=1
				#row = row + bitv

		if d[8]=="A":
			featureA.append(row) 
		else:
			featureB.append(row) 

	return featureA,featureB

def getLabels(data):
	labelA = []
	labelB = []
	for d in data:
		label = -2
		if d[-1]=="Yes":
			label = 1	
		if d[-1]=="No":
			label = -1
		if d[-1]=="Maybe":
			label = 0

		if d[8]=="A":
			labelA.append(label)
		else:
			labelB.append(label)

	return labelA,labelB

def getID(data):
	idsa = []
	idsb = []
	for d in data:
		if d[8]=="A":
			idsa.append(int(d[0]))
		else:
			idsb.append(int(d[0]))

	return idsa,idsb

def rulesA(x):
	price = 0
	sales = 1
	cs1 = 2
	pc1 = 3
	pc2 = 4
	pc3 = 5
	pc4 = 6
	pattr = 7
	swt = 8
	pcost = 9

	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>=195 and x[pc3]<=209):
		return 0
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>332):
		return 1
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>6230):
		return 1

	return -2

def rulesB(x):
	price = 0
	sales = 1
	cs1 = 2
	pc1 = 3
	pc2 = 4
	pc3 = 5
	pc4 = 6
	pattr = 7
	swt = 8
	pcost = 9

	if (x[pc2]==24 or x[pc2]==25) and (x[swt]>66.01):
		return 0
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>332) and (x[swt]>25):
		return 1

	return -2

def rules(x):
	price = 0
	sales = 1
	cs1 = 2
	pc1 = 3
	pc2 = 4
	pc3 = 5
	pc4 = 6
	pattr = 7
	swt = 8
	pcost = 9

	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2387 and x[pc4]<=2514) and (x[pc3]>=210 and x[pc3]<=249):
		return 0
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=2514 and x[pc4]<=6230) and (x[pc3]>209):
		return 1
	if (x[pc2]==24 or x[pc2]==25) and (x[pc4]>=1725 and x[pc4]<=2387):
		return 0
	if (x[pc2]<=22) and (x[pc1]>=2 and x[pc1]<=10) and (x[pattr]>=14 and x[pattr]<=120) and (x[pc3]>332):
		return 1
	if (x[pc2]<=22) and (x[pc1]>=2 and x[pc1]<=10) and (x[pattr]>270):
		return 0
	if (x[pc2]>=22 and x[pc2]<=24):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]>=25 and x[swt]<=39.75) and (x[pcost]<=20.16):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]>39.75) and (x[price]>35.85):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]>=10 and x[swt]<=25) and (x[pattr]>=14 and x[pattr]<=270):
		return -1
	if (x[pc2]<=22) and (x[pc1]<=2) and (x[swt]<=10) and (x[sales]>=18.77 and x[pattr]<=59.47):
		return -1
	return -2


class ElectronicPartsClassification:
	def classifyParts(self,trndata,testdata):
		global trainx,trainy

		delimtrn = [x.split(',')[:-1] for x in trndata]
		delimtest = [x.split(',') for x in testdata]
		idsA,idsB = getID(delimtest)

		scalerA = StandardScaler()
		scalerB = StandardScaler()

		trainxA,trainxB = convertToFeatures(delimtrn)
		trainxA = scalerA.fit_transform(trainxA)
		trainxB = scalerB.fit_transform(trainxB)
		print(len(trainxA[0]))
		testA,testB = convertToFeatures(delimtest)
		testA = scalerA.transform(testA)
		testB = scalerB.transform(testB)

		delimtrn = [x.split(',') for x in trndata]
		trainyA,trainyB = getLabels(delimtrn)
		clf = ExtraTrees(n_estimators=15,subsample=8,min_samples_split=16)
		#clf = BoostedTree([-1,0,1],n_estimators=10,max_depth=-1,max_features=3,min_samples_split=4,learning_rate=0.1,subsample=8)
		#clf = BoostedTreeSimple(n_estimators=2,max_depth=-1,max_features=3,min_samples_split=4,subsample=6)
		#clf = BinaryDecisionTree()
		trainx = trainxA
		trainy = trainyA
		clf.fit(trainx,trainy)
		predA = clf.predict(testA)		
		clf = ExtraTrees(n_estimators=15,subsample=8,min_samples_split=16)
		#clf = ExtraTreesClassifier(n_estimators=20,max_depth=20,min_samples_split=10,bootstrap=True)
		#clf = BoostedTreeSimple(n_estimators=2,max_depth=-1,max_features=3,min_samples_split=4,subsample=6)
		#clf = BoostedTree([-1,0,1],n_estimators=10,max_depth=-1,max_features=3,min_samples_split=4,learning_rate=0.1,subsample=8)
		trainx = trainxB
		trainy = trainyB
		clf.fit(trainx,trainy)
		predB = clf.predict(testB)		

		productidpreds = {}
		for i in range(len(testA)):
			pred = predA[i]

			if idsA[i] not in productidpreds:
				productidpreds[idsA[i]] = {}

			if "A" not in productidpreds[idsA[i]]:
				productidpreds[idsA[i]]["A"] = [pred]
			else:
				productidpreds[idsA[i]]["A"].append(pred)

		for i in range(len(testB)):
			pred = predB[i]

			if idsB[i] not in productidpreds:
				productidpreds[idsB[i]] = {}

			if "B" not in productidpreds[idsB[i]]:
				productidpreds[idsB[i]]["B"] = [pred]
			else:
				productidpreds[idsB[i]]["B"].append(pred)

		predictions = []
		for k,v in productidpreds.iteritems():
			strpred = str(k)+","
			if "A" in v:
				cnt = Counter(v["A"])
				pred = cnt.most_common(n=1)[0][0]
				if pred==-1:
					strpred+="No,"
				elif pred==1:
					strpred+="Yes,"
				else:
					strpred+="Maybe,"
			else:
					strpred+="NA,"
			if "B" in v:
				cnt = Counter(v["B"])
				pred = cnt.most_common(n=1)[0][0]
				if pred==-1:
					strpred+="No,"
				elif pred==1:
					strpred+="Yes,"
				else:
					strpred+="Maybe,"
			else:
					strpred+="NA"

			predictions.append(strpred)

		print(predictions)
		return predictions

def mymain():
	trainingData = []
	testingData = []
	testingTruth = []
	testType = 0
	index=1 
  # read data
	uids = []
	truepredictions = {}
	with open('example_data.csv', 'r') as f:
		header = True
		for line in f:
			# skip header
			if header:
				header = False
				continue
			# remove carriage return
			line = line.rstrip('\n').rstrip('\r')
			pid = line.split(",")[0]
			if pid not in uids:
				uids.append(pid)
	
	trainids = random.sample(uids,2*len(uids)/3)
	'''
	trainids = []
	for i in range(len(uids)):
	    if i%3!=0:
		trainids.append(uids[i])
	'''
	print(len(trainids),len(uids))
	print(trainids)
	with open('example_data.csv', 'r') as f:
		header = True
		for line in f:
			# skip header
			if header:
				header = False
				continue
			# remove carriage return
			line = line.rstrip('\n').rstrip('\r')
			# affect data to training or testing randomly
		
			pid = line.split(",")[0]	
			#if numpy.random.randint(0, 3) == 0 :
			if pid not in trainids:
				# remove the last column
				pos = line.rfind(',')
				testingData.append(line[:pos])
				label = line.split(",")[-1]
				segment = line.split(",")[8]

				if pid not in truepredictions:
					truepredictions[pid] = {}
				truepredictions[pid][segment] = label
			else :
				trainingData.append(line)
			index+=1

	# DemographicMembership instance and predict function call      
	epc = ElectronicPartsClassification()
	testingPred = epc.classifyParts(trainingData, testingData)

	validelements = 0
	totalerror = 0
	for p in testingPred:
		pid = p.split(",")[0]
		alabel = p.split(",")[1]
		blabel = p.split(",")[2]

		if "A" in truepredictions[pid]:
			validelements+=1
			if truepredictions[pid]["A"]=="Maybe" and alabel=="No":
				totalerror+=0.2
			if truepredictions[pid]["A"]=="Yes" and alabel=="No":
				totalerror+=0.7
			if truepredictions[pid]["A"]=="No" and alabel=="Maybe":
				totalerror+=0.5
			if truepredictions[pid]["A"]=="Yes" and alabel=="Maybe":
				totalerror+=0.01
			if truepredictions[pid]["A"]=="No" and alabel=="Yes":
				totalerror+=1.0
			if truepredictions[pid]["A"]=="Maybe" and alabel=="Yes":
				totalerror+=0.01
		if "B" in truepredictions[pid]:
			validelements+=1
			if truepredictions[pid]["B"]=="Maybe" and blabel=="No":
				totalerror+=0.2
			if truepredictions[pid]["B"]=="Yes" and blabel=="No":
				totalerror+=0.7
			if truepredictions[pid]["B"]=="No" and blabel=="Maybe":
				totalerror+=0.5
			if truepredictions[pid]["B"]=="Yes" and blabel=="Maybe":
				totalerror+=0.01
			if truepredictions[pid]["B"]=="No" and blabel=="Yes":
				totalerror+=1.0
			if truepredictions[pid]["B"]=="Maybe" and blabel=="Yes":
				totalerror+=0.01
		if alabel=="NA" and "A" in truepredictions[pid]:
			print("predicted NA for valid A")
		if blabel=="NA" and "B" in truepredictions[pid]:
			print("predicted NA for valid B")

	print(totalerror,validelements,totalerror/validelements)
	score = float(1000000.0 * (1.0 - (totalerror/validelements)))
	print("Score:",score)
	print(len(testingPred),len(truepredictions))
	return score
if __name__ == '__main__':
	avgscore = 0.0	
	for i in range(30):
		print(i)
		avgscore+=mymain()
	avgscore/=30
	print("Avg:",avgscore)
