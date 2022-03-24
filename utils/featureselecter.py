import numpy as np

import sklearn.ensemble as sk_en
import sklearn.naive_bayes as sk_nb
import sklearn.linear_model as sk_lm
import sklearn.model_selection as sk_ms
import sklearn.feature_selection as sk_fs
import xgboost as xgb
import eli5


import numpy as np
import pandas as pd
import random

import sklearn.metrics as sk_m
import sklearn.ensemble as sk_en
import sklearn.naive_bayes as sk_nb
import sklearn.linear_model as sk_lm
import sklearn.model_selection as sk_ms
import sklearn.feature_selection as sk_fs
import xgboost as xgb
from biofes import biplot, feature
import eli5


class FeatureSelecter:
	def __init__(self,method='skb',clf=None, n_vars = None):
		self.n_vars = n_vars
		self.method = method
		#print("Metodo elegido:", self.method)
		if(method=='sfm'):
			if(clf==None):
				self.clf = sk_en.RandomForestClassifier(n_estimators = 100,  max_features = 'auto')
			else:
				self.clf = clf
		elif(self.method=='rfo'):
			if(clf==None):
				self.clf = sk_nb.GaussianNB()
			else:
				self.clf = clf
		elif(self.method=='rfs'):
			if(clf==None):
				self.clf = sk_nb.GaussianNB()
			else:
				self.clf = clf
		elif(self.method=='skb'):
			if(self.n_vars is None):
				self.n_vars=10
			self.clf = sk_fs.SelectKBest(score_func=sk_fs.f_classif, k=self.n_vars)
		elif(self.method=='eli5_rfe'):
			if(self.n_vars is None):
				self.n_vars=10
			if(clf==None):
				base_clf = sk_lm.LogisticRegression()
			else:
				base_clf = clf
			eli5_estimator = eli5.sklearn.PermutationImportance(base_clf, cv=10)
			self.clf = sk_fs.RFE(eli5_estimator, n_features_to_select=self.n_vars, step=1)
		elif(self.method == 'biofes'):
			pass
	def transform(self,X):
		if(self.method=='sfm'):
			if(self.n_vars is None):
				return sk_fs.SelectFromModel(self.clf, prefit=True).transform(X)
			else:
				return sk_fs.SelectFromModel(self.clf, max_features=self.n_vars, threshold=-np.inf, prefit=True).transform(X)
		elif(self.method=='rfo'):
			return X[self.X_columns]
		elif(self.method=='rfs'):
			return X[self.X_columns]
		elif(self.method=='PCA'):
			return self.clf.transform(X)
		elif(self.method=='skb'):
			return self.clf.transform(X)
		elif(self.method=='eq'):
			return X
		elif(self.method=='eli5_rfe'):
			return self.clf.transform(X)
		elif(self.method == 'biofes'):
			var_sel = list(self.Tcan.Disc.sort_values(by='0-1', ascending = False).iloc[:].index)
			return X[var_sel[0:min(self.n_vars, len(var_sel))]]
	def fit(self,X,y):
		if(self.method=='eq'):
			return self
		elif(self.method=='sfm'):
			self.clf = self.clf.fit(X,y)
		elif(self.method=='rfo'):
			self.X_columns = recursive_feature_ordering(X,y, 5, self.n_vars, self.clf)
		elif(self.method=='rfs'):
			self.X_columns = random_fs(X,y,self.n_vars,3,self.clf)
		elif(self.method=='skb'):
			return self.clf.fit(X,y)
		elif(self.method=='eli5_rfe'):
			return self.clf.fit(X,y)
		elif(self.method == 'biofes'):
			self.bip_can = biplot.Canonical(data = X.astype(float), dim = min(35,int(len(X.columns)/2), int(len(X.index)/2)), GroupNames = list(np.unique(y.iloc[:,0])), y = list(y.iloc[:,0]), method = None)
			self.Tcan = feature.selection(self.bip_can, y.iloc[:,0], thr_dis = 40, thr_corr = 0.89)
		return self
	def set_params(self, method, clf=None, n_vars = None):
		self.method=method
		self.n_vars=n_vars
		#print("Metodo elegido:", self.method)
		if(method=='sfm'):
			if(clf==None):
				self.clf = sk_en.RandomForestClassifier(n_estimators = 100,  max_features = 'auto')
			else:
				self.clf = clf
		elif(self.method=='rfo'):
			if(clf==None):
				self.clf = sk_nb.GaussianNB()
			else:
				self.clf = clf
		elif(self.method=='rfs'):
			if(clf==None):
				self.clf = sk_nb.GaussianNB()
			else:
				self.clf = clf
		elif(self.method=='skb'):
			if(self.n_vars is None):
				self.n_vars=10
			self.clf = sk_fs.SelectKBest(score_func=sk_fs.f_classif, k=self.n_vars)
		elif(self.method=='eli5_rfe'):
			if(self.n_vars is None):
				self.n_vars=10
			if(clf==None):
				base_clf = sk_lm.LogisticRegression()
			else:
				base_clf = clf
			eli5_estimator = eli5.sklearn.PermutationImportance(base_clf, cv=10)
			self.clf = sk_fs.RFE(eli5_estimator, n_features_to_select=self.n_vars, step=1)
		elif(self.method == 'biofes'):
			pass

def recursive_feature_ordering(X,Y, n_reps, n_vars, clf):
	X=pd.DataFrame(X)
	Y=pd.DataFrame(Y)
	Yy=Y.astype(bool).values.ravel()

	X_columns = list(X.columns)
	seed = 1

	bestscore=0

	for j in range(0, n_reps):
		sc_columns=[]
		score=0.5


		for i in range(1,len(X_columns)+1):
			var_sel= X_columns[0:i]
			score_old = score

			probas = []
			respuestas =[]
			for j in range(0,10):
				skf = sk_ms.KFold(n_splits=10, random_state=j, shuffle=True)
				y_prob = sk_ms.cross_val_predict(clf, X[var_sel], y=Yy, cv=skf, method='predict_proba')
				probas+=list(y_prob[:,1])
				respuestas+=list(Y.values)
			probas=pd.Series(probas).fillna(0).tolist()
			score = sk_m.roc_auc_score(respuestas,probas)
			sc_columns.append(score-score_old)

		keydict = dict(zip(X_columns, sc_columns))
		X_columns.sort(key=keydict.get, reverse=True)

	return X_columns[0:n_vars]

def random_fs(X,Y,n_vars,n_reps,clf):
	X=pd.DataFrame(X)
	Y=pd.DataFrame(Y)
	Yy=Y.astype(bool).values.ravel()

	variables = list(X.columns)
	splits = int(len(variables)/n_vars)+1

	best_score = 0
	best_var_sel = []

	var_scores = {}
	var_times = {}
	for var in variables:
		var_scores[var]=0
		var_times[var] = 0

	for i in range(0, n_reps):
		random.shuffle(variables)
		for var_sel in chunkIt(variables, splits):
			probas = []
			respuestas =[]
			for j in range(0,10):
				skf = sk_ms.KFold(n_splits=10, random_state=j, shuffle=True)
				y_prob = sk_ms.cross_val_predict(clf, X[var_sel], y=Yy, cv=skf, method='predict_proba')
				probas+=list(y_prob[:,1])
				respuestas+=list(Y.values)
			probas=pd.Series(probas).fillna(0).tolist()
			score = sk_m.roc_auc_score(respuestas,probas)

			# if(score > best_score):
				# best_score = score
				# best_var_sel = var_sel

			for var in var_sel:
				var_scores[var] += score
				var_times[var] += 1
	sorted_vars = sorted(var_scores.keys(), key= lambda k: var_scores[k], reverse=True)

	return sorted_vars[0:n_vars]				


def chunkIt(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg

	return out
