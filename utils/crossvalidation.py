import numpy as np
import pandas as pd
import sklearn.model_selection as sk_ms
import sklearn.utils as sk_u
import sklearn.calibration as sk_cal
import sklearn.linear_model as sk_lm
import time
from .stratifiedgroupkfold import StratifiedGroupKFold

def external_validation_RS(external_data, label, feature_oddratio):

	X = external_data
	Y = external_data.loc[:,[label]]

	Y_prob = pd.Series(0, index=X.index)
	for feat in feature_oddratio.keys():
		Y_prob += feature_oddratio[feat]*X.loc[:,feat]

	#Saved as a list of lists because of compatibility with predict_kfold
	tl_pp_dict={"true_label":[list(Y.values.flat)], "pred_prob":[list(Y_prob.values.flat)]}

	return tl_pp_dict

def external_validation(external_data, label, features, clf):
	X = external_data.loc[:,features]
	Y = external_data.loc[:,[label]]


	Y_prob = clf.predict_proba(X)[:,1]

	#Saved as a list of lists because of compatibility with predict_kfold
	tl_pp_dict={"true_label":[list(Y.values.flat)], "pred_prob":[list(Y_prob)]}

	return tl_pp_dict

def predict_filter_kfold_ML(data, label, features, filter_function, clf, calibration, seed, cvfolds):

	kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)

	predicted_probability = []
	true_label = []

	for train_index, test_index in kf.split(data):
		data_train, data_test = data.iloc[train_index], data.iloc[test_index]

		X_train = filter_function(data_train).loc[:,features]
		Y_train = filter_function(data_train).loc[:,[label]]

		X_train = X_train.loc[~Y_train[label].isnull()]
		Y_train = Y_train.loc[~Y_train[label].isnull()]

		X_test = data_test.loc[:,features]
		Y_test = data_test.loc[:,[label]]

		if (calibration is None):
			clf.fit(X_train, Y_train.values.ravel().astype(int))
			calibrated_clf = clf
		else:
			if hasattr(clf, 'best_estimator_'):
				clf.fit(X_train, Y_train.values.ravel().astype(int))
				if(calibration == 'isotonic'):
					calibrated_clf= sk_cal.CalibratedClassifierCV(clf.best_estimator_, method='isotonic', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				elif(calibration == 'sigmoid'):
					calibrated_clf = sk_cal.CalibratedClassifierCV(clf.best_estimator_, method='sigmoid', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				else:
					print('Unknown Calibration type')
					raise
			else:
				if(calibration == 'isotonic'):
					calibrated_clf = sk_cal.CalibratedClassifierCV(clf, method='isotonic', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				elif(calibration == 'sigmoid'):
					calibrated_clf = sk_cal.CalibratedClassifierCV(clf, method='sigmoid', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
		try:
			Y_prob = calibrated_clf.predict_proba(X_test)
			predicted_probability.append(Y_prob[:,1])
		except:
			Y_prob = calibrated_clf.decision_function(X_test)
			predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_filter_kfold_RS(data, label, features, filter_function, feature_oddratio, seed, cvfolds):

	kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)

	predicted_probability = []
	true_label = []

	for train_index, test_index in kf.split(data):
		data_train, data_test = data.iloc[train_index], data.iloc[test_index]

		X_train = filter_function(data_train).loc[:,features]
		Y_train = filter_function(data_train).loc[:,[label]]

		X_train = X_train.loc[~Y_train[label].isnull()]
		Y_train = Y_train.loc[~Y_train[label].isnull()]

		X_test = data_test
		Y_test = data_test.loc[:,[label]]

		Y_prob = pd.Series(0, index=X_test.index)
		for feat in feature_oddratio.keys():
			Y_prob += feature_oddratio[feat]*X_test.loc[:,feat]

		predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_kfold_ML(data, label, features, cv_type, clf, calibration, seed, cvfolds):


	X = data.loc[:,features]
	Y = data.loc[:,[label]].astype(bool)

	if(cv_type == 'stratifiedkfold'):
		skf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
	elif(cv_type == 'kfold'):
		skf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in skf.split(X,Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

		if (calibration is None):
			clf.fit(X_train, Y_train.values.ravel().astype(int))
			calibrated_clf = clf
		else:
			if hasattr(clf, 'best_estimator_'):
				clf.fit(X_train, Y_train.values.ravel().astype(int))
				if(calibration == 'isotonic'):
					calibrated_clf= sk_cal.CalibratedClassifierCV(clf.best_estimator_, method='isotonic', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				elif(calibration == 'sigmoid'):
					calibrated_clf = sk_cal.CalibratedClassifierCV(clf.best_estimator_, method='sigmoid', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				else:
					print('Unknown Calibration type')
					raise
			else:
				if(calibration == 'isotonic'):
					calibrated_clf = sk_cal.CalibratedClassifierCV(clf, method='isotonic', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				elif(calibration == 'sigmoid'):
					calibrated_clf = sk_cal.CalibratedClassifierCV(clf, method='sigmoid', cv=10)
					calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
		try:
			Y_prob = calibrated_clf.predict_proba(X_test)
			predicted_probability.append(Y_prob[:,1])
		except:
			Y_prob = calibrated_clf.decision_function(X_test)
			predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict


def predict_kfold_RS(data, label, features, cv_type,  feature_oddratio, seed, cvfolds):

	X = data.loc[:, :]
	Y = data.loc[:,[label]].astype(bool)

	if(cv_type == 'stratifiedkfold'):
		skf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
	elif(cv_type == 'kfold'):
		skf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in skf.split(X,Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
		Y_prob = pd.Series(0, index=X_test.index)
		for feat in feature_oddratio.keys():
			Y_prob += feature_oddratio[feat]*X_test.loc[:,feat]
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_groupkfold_ML(data, label, features, group_label, cv_type, clf, calibration, seed, cvfolds):

	X = data.loc[:,features]
	Y = data.loc[:,[label]].astype(bool)
	G = data.loc[:, group_label]

	if (cv_type == 'stratifiedgroupkfold'):
		gkf = StratifiedGroupKFold(cvfolds, random_state=seed, shuffle=True)
	elif (cv_type == 'groupkfold'):
		X, Y, G = sk_u.shuffle(X,Y,G, random_state=seed)
		gkf = GroupKFold(cvfolds)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in gkf.split(X,Y,G):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
		G_train, G_test = G.iloc[train_index], G.iloc[test_index]

		if (calibration is None):
			try:
				clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
			except:
				clf.fit(X_train, Y_train.values.ravel().astype(int))
			calibrated_clf = clf
		else:
			if hasattr(clf, 'best_estimator_'):
				try:
					clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
				except:
					clf.fit(X_train, Y_train.values.ravel().astype(int))
				if(calibration == 'isotonic'):
					calibrated_clf  = sk_cal.CalibratedClassifierCV(clf.best_estimator_, method='isotonic', cv=10)
					try:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
					except:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				elif(calibration == 'sigmoid'):
					calibrated_clf  = sk_cal.CalibratedClassifierCV(clf.best_estimator_, method='sigmoid', cv=10)
					try:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
					except:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				else:
					print('Unknown Calibration type')
					raise
			else:
				if(calibration == 'isotonic'):
					calibrated_clf  = sk_cal.CalibratedClassifierCV(clf, method='isotonic', cv=10)
					try:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
					except:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
				elif(calibration == 'sigmoid'):
					calibrated_clf  = sk_cal.CalibratedClassifierCV(clf, method='sigmoid', cv=10)
					try:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int), groups=G_train)
					except:
						calibrated_clf.fit(X_train, Y_train.values.ravel().astype(int))
		try:
			Y_prob = calibrated_clf.predict_proba(X_test)
			predicted_probability.append(Y_prob[:,1])
		except:
			Y_prob = calibrated_clf.decision_function(X_test)
			predicted_probability.append(Y_prob)
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_groupkfold_RS(data, label, features, group_label, cv_type, feature_oddratio, seed, cvfolds):

	X = data.loc[:,:]
	Y = data.loc[:,[label]].astype(bool)
	G = data.loc[:, group_label]

	if (cv_type == 'stratifiedgroupkfold'):
		gkf = StratifiedGroupKFold(cvfolds, random_state=seed, shuffle=True)
	elif (cv_type == 'groupkfold'):
		X, Y, G = sk_u.shuffle(X,Y,G, random_state=seed)
		gkf = GroupKFold(cvfolds)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	for train_index, test_index in gkf.split(X,Y,G):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

		Y_prob = pd.Series(0, index=X_test.index)
		for feat in feature_oddratio.keys():
			Y_prob += feature_oddratio[feat]*X_test.loc[:,feat]
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_kfold_refitted_RS(data, label, features, cv_type, feature_oddratio, seed, cvfolds):

	X = data.loc[:, :]
	Y = data.loc[:,[label]].astype(bool)

	if(cv_type == 'stratifiedkfold'):
		skf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
	elif(cv_type == 'kfold'):
		skf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	lr = sk_lm.LogisticRegression(penalty='none', solver = 'saga')

	for train_index, test_index in skf.split(X,Y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
		X_train = X_train.loc[:,list(feature_oddratio.keys())]
		lr.fit(X_train, Y_train.values.ravel().astype(int))
		Y_prob = pd.Series(0, index=X_test.index)
		n_feat=0
		for feat in feature_oddratio.keys():
			Y_prob += lr.coef_[0,n_feat]*X_test.loc[:,feat]
			n_feat+=1
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict

def predict_groupkfold_refitted_RS(data, label, features, group_label, cv_type, feature_oddratio, seed, cvfolds):

	X = data.loc[:,:]
	Y = data.loc[:,[label]].astype(bool)
	G = data.loc[:, group_label]

	if (cv_type == 'stratifiedgroupkfold'):
		gkf = StratifiedGroupKFold(cvfolds, random_state=seed, shuffle=True)
	elif (cv_type == 'groupkfold'):
		X, Y, G = sk_u.shuffle(X,Y,G, random_state=seed)
		gkf = GroupKFold(cvfolds)
	else:
		raise('incompatible crossvalidation type')

	predicted_probability = []
	true_label = []

	lr = sk_lm.LogisticRegression(penalty='none', solver = 'saga')

	for train_index, test_index in gkf.split(X,Y,G):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

		X_train = X_train.loc[:,list(feature_oddratio.keys())]
		lr.fit(X_train, Y_train.values.ravel().astype(int))
		Y_prob = pd.Series(0, index=X_test.index)
		n_feat=0
		for feat in feature_oddratio.keys():
			Y_prob += lr.coef_[0,n_feat]*X_test.loc[:,feat]
			n_feat+=1
		predicted_probability.append(list(Y_prob.values.flat))
		true_label.append(list(Y_test.values.flat))

	tl_pp_dict={"true_label":true_label, "pred_prob":predicted_probability}

	return tl_pp_dict


def refitted_oddratios(data, label, feature_oddratio):
	X = data.loc[:, list(feature_oddratio.keys())]
	Y = data.loc[:,[label]].astype(bool)

	lr = sk_lm.LogisticRegression(penalty='none', solver = 'saga')
	lr.fit(X, Y.values.ravel().astype(int))

	refitted_or = {}

	n_feat=0
	for feat in feature_oddratio.keys():
		refitted_or[feat] = lr.coef_[0,n_feat]
		n_feat+=1

	return refitted_or
