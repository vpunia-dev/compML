from collections import namedtuple, Counter, defaultdict
from math import log
from itertools import groupby
import random
import numpy as np

class RandomDecisionTree(object):

		def __init__(self,targets,max_features=-1,max_depth=-1,min_samples=-1):
				self.root_node = None
				self.max_features = max_features
				self.max_depth = max_depth
				self.min_samples = min_samples
				self.rfeature = random.Random(1)
				self.rfeature2 = random.Random(3)
				self.rvalue = random.Random(2)
				self.targets = targets

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
						default_klass = self.get_most_common_class(training_samples)
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
							target_klass = training_samples[0].klass
							root_node = DecisionTreeLeaf(target_klass)
						elif len(training_samples) <= self.min_samples:
							default_klass = self.get_most_common_class(training_samples)
							root_node = DecisionTreeLeaf(default_klass)
						elif self.max_depth!=-1 and count==self.max_depth:
							default_klass = self.get_most_common_class(training_samples)
							root_node = DecisionTreeLeaf(default_klass)
						else:	
							subspace = random.sample(predicting_features,self.max_features)
							best_feature,feature_value = self.select_best_feature(training_samples,subspace,klasses)
							#best_feature = self.rfeature2.choice(subspace)
							best_feature_values = {s.sample[best_feature] for s in training_samples}
							if len(best_feature_values)==1:
								default_klass = self.get_most_common_class(training_samples)
								root_node = DecisionTreeLeaf(default_klass)
							else:
								#feature_value = self.rvalue.uniform(min(best_feature_values),max(best_feature_values))
								root_node = DecisionTreeNode(best_feature,feature_value)
								lsamples = [s for s in training_samples if s.sample[best_feature] < feature_value]
								rsamples = [s for s in training_samples if s.sample[best_feature] >= feature_value]
								lchild = self.create_decision_tree(lsamples,predicting_features,count+1)
								rchild = self.create_decision_tree(rsamples,predicting_features,count+1)
								root_node[feature_value] = (lchild,rchild)
				return root_node

		@staticmethod
		def get_most_common_class(trainning_samples):
				klasses = [s.klass for s in trainning_samples]
				counter = Counter(klasses)
				k, = counter.most_common(n=1)
				return k[0]

		def select_best_feature(self, samples, features, klasses):
				gain_factors = [(self.gini_impurity(samples, feat, klasses)+tuple([feat]))
												for feat in features]
				gain_factors.sort()
				best_feature_value = gain_factors[0][1]
				best_feature = gain_factors[0][2]
				return best_feature,best_feature_value

		def friedman(self, samples, feature, klasses):
				values = {s.sample[feature] for s in samples}
				p = self.rvalue.uniform(min(values),max(values))
				reg = 200
				leaf = 10

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
				return (gain,p)

		def sklearn_gini(self, samples, feature, klasses):
				values = {s.sample[feature] for s in samples}
				partition = self.rvalue.uniform(min(values),max(values))
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lscore=0;rscore=0;pscore=0
				if len(lclasses)!=0:
					lscore = self.skgini(lclasses)
				if len(rclasses)!=0:
					rscore = self.skgini(rclasses)
				pscore = self.skgini(klasses)

				gini = lscore + rscore - pscore
				return (gini,partition)
		def skgini(self,data):
				score = 0.0
				for t in self.targets:
					pk = float(1.0*data.count(t)/len(data)	)
					score += pk*(1-pk)	
				return score
		def gini_impurity(self, samples, feature, klasses):
				N = len(samples)
				values = {s.sample[feature] for s in samples}
				partition = random.uniform(min(values),max(values))
				lclasses = [s.klass for s in samples if s.sample[feature] < partition]
				rclasses = [s.klass for s in samples if s.sample[feature] >= partition]
				lgini = self.gini(lclasses)
				rgini = self.gini(rclasses)
				parentgini = self.gini(klasses)
				pl = float(1.0*len(lclasses)/len(klasses))
				pr = float(1.0*len(rclasses)/len(klasses))

				return (parentgini - (pl*lgini) - (pr*rgini),partition)

		def information_gain(self, samples, feature, klasses):
				N = len(samples)
				values = [s.sample[feature] for s in samples]
				partition = self.rvalue.uniform(min(values),max(values))
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

class ExtraTrees():
	"""extra trees class"""
	def __init__(self,n_estimators=10,max_features=3,max_depth=-1,min_samples_split=5,subsample=10,targets=[-1,0,1]):
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.subsample = subsample
		self.estimators = []
		self.r = random.Random(4)
		self.targets = targets

	def fit(self,X,y):
		trainingsubseti = range(len(X))
		for n in range(self.n_estimators):
			'''
			trainingsubseti = range(len(X))
			indices = self.r.sample(trainingsubseti,self.subsample*len(trainingsubseti)/10)
			trainxn = [X[i] for i in indices]
			trainyn = [y[i] for i in indices]
			'''
			clf = RandomDecisionTree(max_features=self.max_features,max_depth=self.max_depth,min_samples=self.min_samples_split,targets=self.targets)
			clf.fit(X,y)
			self.estimators.append(clf)

	def predict(self,X):
		xpred = []
		for x in X:
			predictions = []	
			for clf in self.estimators:	
				pred = clf.predict([x])
				predictions.append(pred[0])
			counts = [len(list(group)) for k,group in groupby(predictions)]
			p = [list(group) for k,group in groupby(predictions)]
			maxindex = counts.index(max(counts))
			
			xpred.append(p[maxindex][0])	

		return xpred
