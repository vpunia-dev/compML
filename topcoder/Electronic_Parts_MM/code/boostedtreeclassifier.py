from collections import namedtuple, Counter, defaultdict
from math import log
from itertools import groupby
import random
import numpy as np

class ProbabilisticDecisionTree(object):

		def __init__(self,target,max_features=-1,max_depth=-1,min_samples=-1):
				self.root_node = None
				self.max_features = max_features
				self.max_depth = max_depth
				self.min_samples = min_samples
				self.r = random.Random(1)	
				self.target = target
				self.rfeature = random.Random(1)
				self.rvalue = random.Random(2)
				self.rfeature2 = random.Random(3)

		def fit(self, samples, target):
				training_samples = [TrainingSample(s, t)
														for s, t in zip(samples, target)]
				predicting_features = list(range(len(samples[0])))
				if self.max_features == -1:
					self.max_features = len(predicting_features)
				self.root_node = self.create_decision_tree(training_samples,
																									 predicting_features)

		def predict(self, X, allow_unclassified=False):
				default_klass = 1
				predicted_klasses = []

				for sample in X:
						klass = None
						current_node = self.root_node
						while klass is None:
								if current_node.is_leaf():
										klass = current_node.klass
								else:
										key_value = sample[current_node.feature]
										split_val = current_node.split_val
										if key_value < split_val:
												current_node = current_node[split_val][0]
										else:
												current_node = current_node[split_val][1]
						predicted_klasses.append(klass)
				return predicted_klasses

		def create_decision_tree(self, training_samples, predicting_features,count=0):
				if not predicting_features:
						# No more predicting features
						default_klass = self.get_target_probab(training_samples)
						root_node = DecisionTreeLeaf(default_klass)
				else:
						klasses = [sample.klass for sample in training_samples]
						all_same = True 
						for f in predicting_features:
							for s1,s2 in zip(training_samples,training_samples[1:]):
								if s1.sample[f] != s2.sample[f]:
									all_same = False
									break	
						if len(set(klasses)) == 1:
							target_klass = self.get_target_probab(training_samples)
							root_node = DecisionTreeLeaf(target_klass)
						if self.max_depth!=-1 and count==self.max_depth:
							target_klass = self.get_target_probab(training_samples)
							root_node = DecisionTreeLeaf(target_klass)
						else:
							best_feature = 0
							feature_value = 0	
							if self.max_features!=-1:
								subspace = self.rfeature.sample(predicting_features,self.max_features)
								best_feature = self.rfeature2.choice(subspace)
								#subspace = self.r.sample(predicting_features,self.max_features)
								#best_feature,feature_value = self.select_best_feature(training_samples,subspace,klasses)
							else:
								best_feature,feature_value = self.select_best_feature(training_samples,predicting_features,klasses)

							best_feature_values = {s.sample[best_feature] for s in training_samples}
							if len(best_feature_values)==1:
								default_klass = self.get_target_probab(training_samples)
								root_node = DecisionTreeLeaf(default_klass)
							else:
								feature_value = self.rvalue.uniform(min(best_feature_values),max(best_feature_values))
								root_node = DecisionTreeNode(best_feature,feature_value)
								lsamples = [s for s in training_samples if s.sample[best_feature] < feature_value]
								rsamples = [s for s in training_samples if s.sample[best_feature] >= feature_value]
								if len(lsamples)==0 or len(rsamples)==0:
									default_klass = self.get_target_probab(training_samples)
									root_node = DecisionTreeLeaf(default_klass)
								else:
									lchild = self.create_decision_tree(lsamples,predicting_features,count+1)
									rchild = self.create_decision_tree(rsamples,predicting_features,count+1)
									root_node[feature_value] = (lchild,rchild)
				return root_node

		def get_target_probab(self,trainning_samples):
				klasses = [s.klass for s in trainning_samples]
				targetcount = klasses.count(self.target)
				return float(1.0*targetcount/len(klasses))

		def select_best_feature(self, samples, features, klasses):
				gain_factors = [(self.gini_impurity(samples, feat, klasses)+tuple([feat]))
												for feat in features]
				gain_factors.sort()
				best_feature_value = gain_factors[-1][1]
				best_feature = gain_factors[-1][2]
				if self.max_features==-1:
					features.pop(features.index(best_feature))
				return best_feature,best_feature_value

		def friedman(self, samples, feature, klasses):
				values = [s.sample[feature] for s in samples]
				values.sort()
				maxgain = -100000000
				partition = 0
				reg = 200
				leaf = 10
				for p in values:
					lclasses = [-s.klass for s in samples if s.sample[feature] < p]
					rclasses = [-s.klass for s in samples if s.sample[feature] >= p]
					gl = sum(lclasses)**2
					gr = sum(rclasses)**2
					hl = 2*len(lclasses)
					hr = 2*len(rclasses)
					leftscore = float(1.0*(gl**2)/(hl+reg))
					rightscore = float(1.0*(gr**2)/(hr+reg))
					parentscore = float(1.0* (gl+gr)**2/(hl+hr+reg))
					
					gain = leftscore + rightscore - parentscore - leaf
					if gain > maxgain:
						maxgain = gain
						partition = p
					else:
						break
				return (maxgain,partition)
		def gini_impurity2(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				values.sort()
				maxgini = -100000000
				partition = 0
				for p in values:
					lclasses = [s.klass for s in samples if s.sample[feature] < p]
					rclasses = [s.klass for s in samples if s.sample[feature] >= p]
					lgini = self.gini(lclasses)
					rgini = self.gini(rclasses)
					parentgini = self.gini(klasses)
					pl = float(1.0*len(lclasses)/len(klasses))
					pr = float(1.0*len(rclasses)/len(klasses))

					gin = (parentgini - (pl*lgini) - (pr*rgini))
					if gin>maxgini:
						maxgini = gin
						partition = p
				return (maxgini,partition)
					
		def gini_impurity(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				partition = np.mean(values)
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lgini = self.gini(lclasses)
				rgini = self.gini(rclasses)
				parentgini = self.gini(klasses)
				pl = float(1.0*len(lclasses)/len(klasses))
				pr = float(1.0*len(rclasses)/len(klasses))

				return (parentgini - (pl*lgini) - (pr*rgini),partition)

		def information_gain2(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				values.sort()
				maxent= -100000000
				partition = 0
				for p in values:
					lclasses = [s.klass for s in samples if s.sample[feature] < p]
					rclasses = [s.klass for s in samples if s.sample[feature] >= p]
					lgini = self.entropy(lclasses)
					rgini = self.entropy(rclasses)
					parentgini = self.entropy(klasses)
					pl = float(1.0*len(lclasses)/len(klasses))
					pr = float(1.0*len(rclasses)/len(klasses))

					ent = parentgini - (pl*lgini) - (pr*rgini)
					if ent>maxent:
						maxent = ent
						partition = p
					else:
						break
				return (maxent,partition)
		def information_gain(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				partition = np.mean(values)
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lentropy = self.entropy(lclasses)
				rentropy = self.entropy(rclasses)
				parentropy = self.entropy(klasses)
				pl = float(1.0*len(lclasses)/len(klasses))
				pr = float(1.0*len(rclasses)/len(klasses))

				return (parentropy - (pl*lentropy) - (pr*rentropy),partition)
				N = len(samples)
				samples_partition = defaultdict(list)
				for s in samples:
						samples_partition[s.sample[feature]].append(s)
				feature_entropy = 0.0
				min_entropy = 100000000
				min_partition = 0
				for value,partition in samples_partition.items():
						sub_klasses = [s.klass for s in partition]
						value_entropy = (len(partition) / N) * self.entropy(sub_klasses)
						feature_entropy += value_entropy
						if value_entropy < min_entropy:
							min_entropy = value_entropy
							min_partition = value

				return (self.entropy(klasses) - feature_entropy,min_partition)

		@staticmethod
		def gini(dataset):
				N = len(dataset)
				counter = Counter(dataset)
				return 1.0 - sum(-1.0 * ((counter[k]/N)**2) for k in counter)

		@staticmethod
		def entropy(dataset):
				N = len(dataset)
				counter = Counter(dataset)
				return sum(-1.0*(counter[k] / N)*log( float(1.0*counter[k] / N),2) for k in counter)

class RegressionDecisionTree(object):

		def __init__(self,max_features=-1,max_depth=-1,min_samples=-1):
				self.root_node = None
				self.max_features = max_features
				self.max_depth = max_depth
				self.min_samples = min_samples
				self.r = random.Random(1)	
				self.rfeature = random.Random(1)
				self.rvalue = random.Random(2)
				self.rfeature2 = random.Random(3)

		def fit(self, samples, target):
				training_samples = [TrainingSample(s, t)
														for s, t in zip(samples, target)]
				predicting_features = list(range(len(samples[0])))
				if self.max_features == -1:
					self.max_features = len(predicting_features)
				self.root_node = self.create_decision_tree(training_samples,
																									 predicting_features)

		def predict(self, X, allow_unclassified=False):
				default_klass = 1
				predicted_klasses = []

				for sample in X:
						klass = None
						current_node = self.root_node
						while klass is None:
								if current_node.is_leaf():
										klass = current_node.klass
								else:
										key_value = sample[current_node.feature]
										split_val = current_node.split_val
										if key_value < split_val:
												current_node = current_node[split_val][0]
										else:
												current_node = current_node[split_val][1]
						predicted_klasses.append(klass)
				return predicted_klasses

		def score(self, X, target, allow_unclassified=True):
				predicted = self.predict(X, allow_unclassified=allow_unclassified)
				n_matches = sum(p == t for p, t in zip(predicted, target))
				return 1.0 * n_matches / len(X)

		def create_decision_tree(self, training_samples, predicting_features,count=0):
				if not predicting_features:
						# No more predicting features
						default_klass = self.get_target_probab(training_samples)
						root_node = DecisionTreeLeaf(default_klass)
				else:
						klasses = [sample.klass for sample in training_samples]
						all_same = True 
						for f in predicting_features:
							for s1,s2 in zip(training_samples,training_samples[1:]):
								if s1.sample[f] != s2.sample[f]:
									all_same = False
									break	
						if len(set(klasses)) == 1:
							target_klass = self.get_target_probab(training_samples)
							root_node = DecisionTreeLeaf(target_klass)
						if self.max_depth!=-1 and count == self.max_depth:
							target_klass = self.get_target_probab(training_samples)
							root_node = DecisionTreeLeaf(target_klass)
						else:	
							best_feature = 0
							feature_value = 0 
							if self.max_features!=-1:
								subspace = self.rfeature.sample(predicting_features,self.max_features)
								best_feature = self.rfeature2.choice(subspace)
								#subspace = self.r.sample(predicting_features,self.max_features)
								#best_feature,feature_value = self.select_best_feature(training_samples,subspace,klasses)
							else:
								best_feature,feature_value = self.select_best_feature(training_samples,predicting_features,klasses)

							best_feature_values = {s.sample[best_feature] for s in training_samples}
							if len(best_feature_values)==1:
								default_klass = self.get_target_probab(training_samples)
								root_node = DecisionTreeLeaf(default_klass)
							els e:
								feature_value = self.rvalue.uniform(min(best_feature_values),max(best_feature_values))
								root_node = DecisionTreeNode(best_feature,feature_value)
								lsamples = [s for s in training_samples if s.sample[best_feature] < feature_value]
								rsamples = [s for s in training_samples if s.sample[best_feature] >= feature_value]
								if len(lsamples)==0 or len(rsamples)==0:
									default_klass = self.get_target_probab(training_samples)
									root_node = DecisionTreeLeaf(default_klass)
								else:
									lchild = self.create_decision_tree(lsamples,predicting_features,count+1)
									rchild = self.create_decision_tree(rsamples,predicting_features,count+1)
									root_node[feature_value] = (lchild,rchild)
				return root_node

		def get_target_probab(self,trainning_samples):
				klasses = np.array([s.klass for s in trainning_samples])
				return np.mean(klasses) 

		def select_best_feature(self, samples, features, klasses):
				gain_factors = [(self.gini_impurity(samples, feat, klasses)+tuple([feat]))
												for feat in features]
				gain_factors.sort()
				best_feature_value = gain_factors[-1][1]
				best_feature = gain_factors[-1][2]
				if self.max_features==-1:
					features.pop(features.index(best_feature))
				return best_feature,best_feature_value

		def friedman(self, samples, feature, klasses):
				values = [s.sample[feature] for s in samples]
				values.sort()
				maxgain = -100000000
				partition = 0
				reg = 200
				leaf = 10
				for p in values:
					lclasses = [-s.klass for s in samples if s.sample[feature] < p]
					rclasses = [-s.klass for s in samples if s.sample[feature] >= p]
					gl = sum(lclasses)**2
					gr = sum(rclasses)**2
					hl = 2*len(lclasses)
					hr = 2*len(rclasses)
					leftscore = float(1.0*(gl**2)/(hl+reg))
					rightscore = float(1.0*(gr**2)/(hr+reg))
					parentscore = float(1.0* (gl+gr)**2/(hl+hr+reg))
					
					gain = leftscore + rightscore - parentscore - leaf
					if gain > maxgain:
						maxgain = gain
						partition = p
					else:
						break
				return (maxgain,partition)
					
		def gini_impurity2(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				values.sort()
				maxgini = -100000000
				partition = 0
				for p in values:
					lclasses = [s.klass for s in samples if s.sample[feature] < p]
					rclasses = [s.klass for s in samples if s.sample[feature] >= p]
					lgini = self.gini(lclasses)
					rgini = self.gini(rclasses)
					parentgini = self.gini(klasses)
					pl = float(1.0*len(lclasses)/len(klasses))
					pr = float(1.0*len(rclasses)/len(klasses))

					gin = (parentgini - (pl*lgini) - (pr*rgini))
					if gin>maxgini:
						maxgini = gin
						partition = p
				return (maxgini,partition)
		def gini_impurity(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				partition = np.mean(values)
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lgini = self.gini(lclasses)
				rgini = self.gini(rclasses)
				parentgini = self.gini(klasses)
				pl = float(1.0*len(lclasses)/len(klasses))
				pr = float(1.0*len(rclasses)/len(klasses))

				return (parentgini - (pl*lgini) - (pr*rgini),partition)

		def information_gain2(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				values.sort()
				maxent= -100000000
				partition = 0
				for p in values:
					lclasses = [s.klass for s in samples if s.sample[feature] < p]
					rclasses = [s.klass for s in samples if s.sample[feature] >= p]
					lgini = self.entropy(lclasses)
					rgini = self.entropy(rclasses)
					parentgini = self.entropy(klasses)
					pl = float(1.0*len(lclasses)/len(klasses))
					pr = float(1.0*len(rclasses)/len(klasses))

					ent = parentgini - (pl*lgini) - (pr*rgini)
					if ent>maxent:
						maxent = ent
						partition = p
					else:
						break
				return (maxent,partition)

		def information_gain(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				partition = np.mean(values)
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lentropy = self.entropy(lclasses)
				rentropy = self.entropy(rclasses)
				parentropy = self.entropy(klasses)
				pl = float(1.0*len(lclasses)/len(klasses))
				pr = float(1.0*len(rclasses)/len(klasses))

				return (parentropy - (pl*lentropy) - (pr*rentropy),partition)

		@staticmethod
		def gini(dataset):
				N = len(dataset)
				counter = Counter(dataset)
				return 1.0 - sum(-1.0 * ((counter[k]/N)**2) for k in counter)

		@staticmethod
		def entropy(dataset):
				N = len(dataset)
				counter = Counter(dataset)
				return sum(-1.0*(counter[k] / N)*log( float(1.0*counter[k] / N),2) for k in counter)
TrainingSample = namedtuple('TrainingSample', ('sample', 'klass'))


class DecisionTreeNode(dict):
    def __init__(self, feature, split_val = -1,*args, **kwargs):
        self.feature = feature
        self.split_val = split_val 
        super(DecisionTreeNode, self).__init__(*args, **kwargs)

    def is_leaf(self):
        return False

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.feature)


class DecisionTreeLeaf(dict):
    def __init__(self, klass, *args, **kwargs):
        self.klass = klass
        super(DecisionTreeLeaf, self).__init__(*args, **kwargs)

    def is_leaf(self):
        return True

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.klass)
class BoostedTree():
	def __init__(self,classes,n_estimators=10,max_features=10,max_depth=10,min_samples_split=10,subsample=7,learning_rate=0.1):
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.subsample = subsample
		self.estimators = []
		self.r = random.Random(200)
		self.classes = classes
		self.models = defaultdict(list)
		self.modelsdata = {}
		self.modelstrue = {}
		self.learning_rate = learning_rate
	def fit(self,X,y):
		for c in self.classes:
			classy = [0 if y[i]!=c else 1 for i in range(len(y))]
			self.modelsdata[c] = classy
			self.modelstrue[c] = classy

		for c in self.classes:
			trainx = X
			trainy = self.modelsdata[c]

			sampleindices = self.r.sample(range(len(X)),self.subsample*len(X)/10)
			samplex = [trainx[i] for i in sampleindices]
			sampley = [trainy[i] for i in sampleindices]

			clf = ProbabilisticDecisionTree(c,max_depth=-1,max_features=self.max_features)
			clf.fit(samplex,sampley)
			self.models[c].append(clf)
			preds = np.array([0]*len(X))
			for est in self.models[c]:
				preds = preds + np.array(est.predict(X))
			error = [(self.modelsdata[c][i]-preds[i]) for i in range(len(preds))]
			self.modelsdata[c] = error

		for n in range(self.n_estimators):
			for c in self.classes:
				trainx = X
				trainy = self.modelsdata[c]

				sampleindices = self.r.sample(range(len(X)),self.subsample*len(X)/10)
				samplex = [trainx[i] for i in sampleindices]
				sampley = [trainy[i] for i in sampleindices]

				clf = RegressionDecisionTree(max_depth=self.max_depth,max_features=self.max_features)
				clf.fit(samplex,sampley)
				self.models[c].append(clf)
				preds = np.array([0]*len(X))
				for est in self.models[c]:
					preds = preds + self.learning_rate*np.array(est.predict(X))
				error = [(self.modelsdata[c][i]-preds[i]) for i in range(len(preds))]
				self.modelsdata[c] = error

	def predict(self,X):
		xpred = []
		for x in X:
			pred = 0
			origpred = -100000000
			label = -2
			for c in self.classes:
				pred = 0
				for clf in self.models[c]:
					pred += clf.predict([x])[0]
				if pred > origpred:
					origpred = pred	
					label = c
			xpred.append(label)

		return xpred
