import matplotlib.pyplot as plt
import sklearn.metrics as sk_m
import scipy.stats as sc_st
import scipy.misc as sc_ms
import pandas as pd
import numpy as np
import pickle
import glob
import eli5
import os

from user_variables_info import dict_var
from user_MLmodels_info import ML_info
from user_RiskScores_info import RS_info

def create_descriptive_xls(data, wf_name, label_name):

	j=label_name
	row_list =[]

	if (len(dict_var.keys())==0):
		for i in data.columns:
			if(data[i].dtype in ['float64','float32','int64','int32','bool']):
				dict_var[i]=i

	for i in data.columns:
		if i in dict_var.keys():
			if(sorted(list(data.loc[data[i].notnull(),i].unique())) == [0,1]):
				t0 = list(data[i]).count(0)
				t1 = list(data[i]).count(1)
				f00 = list(data.loc[data[j]==0,i]).count(0)
				f01 = list(data.loc[data[j]==0,i]).count(1)
				f10 = list(data.loc[data[j]==1,i]).count(0)
				f11 = list(data.loc[data[j]==1,i]).count(1)
				pvalue = sc_st.fisher_exact([[f00,f01],[f10,f11]])[1]

				dt = data[i].astype(float).describe()
				d0 = data.loc[data[j]==0,i].astype(float).describe()
				d1 = data.loc[data[j]==1,i].astype(float).describe()
				row = {'Name':dict_var[i], 'N':(t0+t1), 'Mean':f"{t1} ({dt['mean']:.1%})",
					  j+'_0_N':d0['count'],j+'_0_Mean':f"{f01} ({d0['mean']:.1%})",
					  j+'_1_N':d1['count'],j+'_1_Mean':f"{f11} ({d1['mean']:.1%})",
					  'p-value':f'{pvalue:.3f}'}
				row_list.append(row)

			else:
				pvalue = sc_st.ttest_ind(data.loc[data[j]==0, i].astype(float),
										  data.loc[data[j]==1, i].astype(float),
										  nan_policy='omit')[1]

				dt = data[i].astype(float).describe()
				d0 = data.loc[data[j]==0, i].astype(float).describe()
				d1 = data.loc[data[j]==1, i].astype(float).describe()

				row = {'Name':dict_var[i], 'N':dt['count'], 'Mean':f'{dt["mean"]:.1f}±{dt["std"]:.1f}',
					   j+'_0_N':d0['count'], j+'_0_Mean':f'{d0["mean"]:.1f}±{d0["std"]:.1f}',
					   j+'_1_N':d1['count'], j+'_1_Mean':f'{d1["mean"]:.1f}±{d1["std"]:.1f}',
					   'p-value':f'{pvalue:.3f}'}
				row_list.append(row)


	if(len(row_list) > 0):
		df_temp = pd.DataFrame(row_list).set_index('Name')

		indices = [x for x in dict_var.values() if x in df_temp.index]
		columnas=['N', 'Mean', j+'_0_N', j+'_0_Mean',j+'_1_N',j+'_1_Mean','p-value']

		df_temp = df_temp.loc[indices,columnas]
		return df_temp
	else:
		return -1

def cutoff_threshold_single(y_prob, y_true):
	#This functions searches for the best threshold to maximize AUC with a single point
	y_true = np.array(y_true)
	y_prob = np.array(y_prob)

	best_threshold=0
	max_auc=0
	fpr,tpr, thresholds = sk_m.roc_curve(y_true, y_prob)

	if (len(thresholds) > 100) :
		new_thresholds=[]
		new_fpr=[]
		new_tpr=[]
		list_subarrays_thr=np.array_split(thresholds,100)
		list_subarrays_tpr=np.array_split(tpr,100)
		list_subarrays_fpr=np.array_split(fpr,100)
		for i in list_subarrays_thr:
			new_thresholds.append(i[0])
		for i in list_subarrays_fpr:
			new_fpr.append(i[0])
		for i in list_subarrays_tpr:
			new_tpr.append(i[0])

		thresholds=np.array(new_thresholds)
		fpr = np.array(new_fpr)
		tpr = np.array(new_tpr)

	for (x1, y1, t1) in zip(fpr, tpr, thresholds):
		auc = sk_m.auc([0,x1,1],[0,y1,1])
		if (auc > max_auc):
			max_auc=auc
			best_threshold=t1

	(tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = contingency_table_calculator(y_true, y_prob, best_threshold)

	return (best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv)


def cutoff_threshold_double(y_prob, y_true):
	#This functions searches for the best threshold to maximize AUC with a single point
	y_true = np.array(y_true)
	y_prob = np.array(y_prob)

	best_threshold1=0
	best_threshold2=1
	max_auc=0
	fpr,tpr, thresholds = sk_m.roc_curve(y_true, y_prob)

	if (len(thresholds) > 100) :
		new_thresholds=[]
		new_fpr=[]
		new_tpr=[]
		list_subarrays_thr=np.array_split(thresholds,100)
		list_subarrays_tpr=np.array_split(tpr,100)
		list_subarrays_fpr=np.array_split(fpr,100)
		for i in list_subarrays_thr:
			new_thresholds.append(i[0])
		for i in list_subarrays_fpr:
			new_fpr.append(i[0])
		for i in list_subarrays_tpr:
			new_tpr.append(i[0])

		thresholds=np.array(new_thresholds)
		fpr = np.array(new_fpr)
		tpr = np.array(new_tpr)

	for (x1, y1, t1) in zip(fpr, tpr, thresholds):
		for (x2, y2, t2) in zip(fpr[thresholds > t1], tpr[thresholds>t1], thresholds[thresholds>t1]):
			if(x2 > x1):
				auc = sk_m.auc([0,x1,x2,1],[0,y1,y2,1])
			else:
				auc = sk_m.auc([0,x2,x1,1],[0,y2,y1,1])
			if (auc > max_auc):
				max_auc=auc
				best_threshold1=t1
				best_threshold2=t2

	(tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = contingency_table_calculator(y_true, y_prob, best_threshold1)
	(tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = contingency_table_calculator(y_true, y_prob, best_threshold2)

	return {"threshold1":(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1),
			"threshold2":(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2)}

def cutoff_threshold_triple(y_prob, y_true):
	#This functions searches for the best threshold to maximize AUC with a single point
	y_true = np.array(y_true)
	y_prob = np.array(y_prob)

	best_threshold1=0
	best_threshold2=0.5
	best_threshold3=1
	max_auc=0
	fpr,tpr, thresholds = sk_m.roc_curve(y_true, y_prob)


	if (len(thresholds) > 100) :
		new_thresholds=[]
		new_fpr=[]
		new_tpr=[]
		list_subarrays_thr=np.array_split(thresholds,100)
		list_subarrays_tpr=np.array_split(tpr,100)
		list_subarrays_fpr=np.array_split(fpr,100)
		for i in list_subarrays_thr:
			new_thresholds.append(i[0])
		for i in list_subarrays_fpr:
			new_fpr.append(i[0])
		for i in list_subarrays_tpr:
			new_tpr.append(i[0])

		thresholds=np.array(new_thresholds)
		fpr = np.array(new_fpr)
		tpr = np.array(new_tpr)


	for (x1, y1, t1) in zip(fpr, tpr, thresholds):
		for (x2, y2, t2) in zip(fpr[thresholds > t1], tpr[thresholds>t1], thresholds[thresholds>t1]):
			for (x3, y3, t3) in zip(fpr[thresholds > t2], tpr[thresholds>t2], thresholds[thresholds>t2]):
				if(x2 > x1):
					auc = sk_m.auc([0,x1,x2,x3,1],[0,y1,y2,y3,1])
				else:
					auc = sk_m.auc([0,x3,x2,x1,1],[0,y3,y2,y1,1])
				if (auc > max_auc):
					max_auc=auc
					best_threshold1=t1
					best_threshold2=t2
					best_threshold3=t3

	(tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = contingency_table_calculator(y_true, y_prob, best_threshold1)
	(tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = contingency_table_calculator(y_true, y_prob, best_threshold2)
	(tprate3, fprate3, tnrate3, fnrate3, sens3, spec3, prec3, nprv3) = contingency_table_calculator(y_true, y_prob, best_threshold3)

	return {"threshold1":(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1),
			"threshold2":(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2),
			"threshold3":(best_threshold3, tprate3, fprate3, tnrate3, fnrate3, sens3, spec3, prec3, nprv3)}


def cutoff_threshold_maxfbeta(y_prob, y_true, beta):
	y_true = np.array(y_true)
	y_prob = np.array(y_prob)

	precision, recall, thresholds = sk_m.precision_recall_curve(y_true, y_prob)

	if (len(thresholds) > 100) :
		new_thresholds=[]
		new_pre=[]
		new_rec=[]
		list_subarrays_thr=np.array_split(thresholds,100)
		list_subarrays_pre=np.array_split(precision,100)
		list_subarrays_rec=np.array_split(recall,100)
		for i in list_subarrays_thr:
			new_thresholds.append(i[0])
		for i in list_subarrays_pre:
			new_pre.append(i[0])
		for i in list_subarrays_rec:
			new_rec.append(i[0])

		thresholds=np.array(new_thresholds)
		precision = np.array(new_pre)
		recall = np.array(new_rec)


	max_f1 = 0
	for r, p, t in zip(recall, precision, thresholds):
		if p + r == 0: continue
		if fbeta(p,r,beta) > max_f1:
			max_f1 = fbeta(p,r,beta)
			max_f1_threshold = t

	(tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = contingency_table_calculator(y_true, y_prob, max_f1_threshold)

	return (max_f1_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv)

def cutoff_threshold_accuracy(y_prob, y_true):
	#This functions searches for the best threshold to maximize AUC with a single point
	y_true = np.array(y_true)
	y_prob = np.array(y_prob)

	fpr,tpr, thresholds = sk_m.roc_curve(y_true, y_prob)

	if (len(thresholds) > 100) :
		new_thresholds=[]
		new_fpr=[]
		new_tpr=[]
		list_subarrays_thr=np.array_split(thresholds,100)
		list_subarrays_tpr=np.array_split(tpr,100)
		list_subarrays_fpr=np.array_split(fpr,100)
		for i in list_subarrays_thr:
			new_thresholds.append(i[0])
		for i in list_subarrays_fpr:
			new_fpr.append(i[0])
		for i in list_subarrays_tpr:
			new_tpr.append(i[0])

		thresholds=np.array(new_thresholds)
		fpr = np.array(new_fpr)
		tpr = np.array(new_tpr)

	best_threshold=0
	true = 0

	for t in thresholds:
		tn = list(y_true[y_prob <  t]).count(0)
		tp = list(y_true[y_prob >= t]).count(1)
		if ((tn+tp) > true):
			true=tn+tp
			best_threshold=t

	(tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = contingency_table_calculator(y_true, y_prob, best_threshold)

	return (best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv)


def all_thresholds(y_prob, y_true):
	y_true = np.array(y_true)
	y_prob = np.array(y_prob)

	fpr,tpr, thresholds = sk_m.roc_curve(y_true, y_prob)

	list_thresholds =[]

	for i in thresholds:
		(tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = contingency_table_calculator(y_true, y_prob, i)
		list_thresholds.append((i, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv))

	return list_thresholds

def contingency_table_calculator(y_true, y_prob, threshold):
	y_true = np.array(y_true)
	y_prob = np.array(y_prob)

	tn = list(y_true[y_prob <  threshold]).count(0)
	fn = list(y_true[y_prob <  threshold]).count(1)
	tp = list(y_true[y_prob >= threshold]).count(1)
	fp = list(y_true[y_prob >= threshold]).count(0)

	if((tp+fn)!=0):
		sens = (tp/(tp+fn))
	else:
		sens = (1)

	if((tn+fp)!=0):
		spec = (tn/(tn+fp))
	else:
		spec = (1)

	if((tp+fp)!=0):
		prec = (tp/(tp+fp))
	else:
		prec = (1)

	if((tn+fn)!=0):
		nprv = (tn/(tn+fn))
	else:
		nprv = (1)

	total = tp + fp + tn + fn
	tprate = tp/total
	fprate = fp/total
	tnrate = tn/total
	fnrate = fn/total

	return (tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv)

def fbeta(p,r,beta):
	if((beta**2*p+r)==0):
		return 0
	return (1+beta**2)*r*p/(beta**2*p+r)

def plot_all_rocs(task_requires, fig_path,title):
	plt.figure(figsize=(10,10))
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Luck', alpha=.8)
	cmap=plt.get_cmap("tab10")
	if(len(task_requires)>10) :
		cmap=plt.get_cmap("tab20")
	color_index=0

	for score in task_requires.keys():
		with open(task_requires[score]["pred_prob"].path, 'rb') as f:
			pred_prob=pickle.load(f)
		with open(task_requires[score]["true_label"].path, 'rb') as f:
			true_label=pickle.load(f)
		with open(task_requires[score]["auc_results"].path, 'rb') as f:
			results_dict=pickle.load(f)

		if(score in ML_info.keys()):
			score_name = ML_info[score]["formal_name"]
		elif(score in RS_info.keys()):
			score_name = RS_info[score]["formal_name"]
		else:
			score_name = "ERROR: Unknown score or classifier"

		pred_prob = pred_prob[~np.isnan(true_label)]
		true_label = true_label[~np.isnan(true_label)]

		fpr, tpr, thresholds = sk_m.roc_curve(true_label,pred_prob)
		plt.plot(fpr, tpr, lw=2, alpha=1, color=cmap(color_index) , label = f'{score_name}: AUC ={results_dict["avg_aucroc"]:1.2f} ({results_dict["aucroc_95ci_low"]:1.2f}-{results_dict["aucroc_95ci_high"]:1.2f})' )
		color_index+=1

	plt.title(title, fontsize=20)
	plt.xlabel('1-specificity', fontsize = 15)
	plt.ylabel('sensitivity', fontsize = 15)
	plt.legend(loc="lower right", fontsize = 15)

	plt.savefig(fig_path)

def plot_all_prs(task_requires, fig_path,title):
	plt.figure(figsize=(10,10))
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Luck', alpha=.8)
	cmap=plt.get_cmap("tab10")
	if(len(task_requires)>10) :
		cmap=plt.get_cmap("tab20")
	color_index=0

	for score in task_requires.keys():
		with open(task_requires[score]["pred_prob"].path, 'rb') as f:
			pred_prob=pickle.load(f)
		with open(task_requires[score]["true_label"].path, 'rb') as f:
			true_label=pickle.load(f)
		with open(task_requires[score]["auc_results"].path, 'rb') as f:
			results_dict=pickle.load(f)

		if(score in ML_info.keys()):
			score_name = ML_info[score]["formal_name"]
		elif(score in RS_info.keys()):
			score_name = RS_info[score]["formal_name"]
		else:
			score_name = "ERROR: Unknown score or classifier"

		with open('./analisis.txt', 'a') as f:
			f.write(task_requires[score]["auc_results"].path)
			f.write('\n')

		pred_prob = pred_prob[~np.isnan(true_label)]
		true_label = true_label[~np.isnan(true_label)]

		prec, recall, thresholds = sk_m.precision_recall_curve(true_label,pred_prob)
		plt.plot(recall, prec, lw=2, alpha=1, color=cmap(color_index) , label = f'{score_name}: AUC ={results_dict["avg_aucpr"]:1.2f} ({results_dict["aucpr_95ci_low"]:1.2f}-{results_dict["aucpr_95ci_high"]:1.2f})' )
		color_index+=1

	plt.title(title, fontsize=20)
	plt.xlabel('recall (sensitivity)', fontsize = 15)
	plt.ylabel('precision', fontsize = 15)
	plt.legend(loc="lower right", fontsize = 15)

	plt.savefig(fig_path)


def group_files_analyze(task_requires, clf_name):
	n_reps=0
	n_repfolds=0
	aucroc_score=0
	aucroc_score2=0
	aucpr_score=0
	aucpr_score2=0
	unfolded_true_label = []
	unfolded_pred_prob = []

	for i in task_requires:
		n_reps+=1
		with open(i.path, 'rb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			rp_dict=pickle.load(f)

		for true_label, pred_prob in zip(rp_dict['true_label'], rp_dict['pred_prob']):
			n_repfolds+=1
			true_label = np.array(true_label)
			pred_prob = np.array(pred_prob)
			repfold_aucroc = sk_m.roc_auc_score(true_label[~np.isnan(true_label)].astype(bool),pred_prob[~np.isnan(true_label)])
			aucroc_score+=repfold_aucroc
			aucroc_score2+=repfold_aucroc**2
			repfold_aucpr = sk_m.average_precision_score(true_label[~np.isnan(true_label)].astype(bool),pred_prob[~np.isnan(true_label)])
			aucpr_score+=repfold_aucpr
			aucpr_score2+=repfold_aucpr**2
			unfolded_true_label+=list(true_label)
			unfolded_pred_prob+=list(pred_prob)

	n_folds=n_repfolds/n_reps

	unfolded_true_label=np.array(unfolded_true_label)
	unfolded_pred_prob =np.array(unfolded_pred_prob)

	pooling_aucroc = sk_m.roc_auc_score(unfolded_true_label[~np.isnan(unfolded_true_label)].astype(bool),unfolded_pred_prob[~np.isnan(unfolded_true_label)])
	averaging_aucroc = aucroc_score/n_repfolds
	averaging_sample_variance_aucroc = (aucroc_score2-aucroc_score**2/n_repfolds)/(n_repfolds-1)

	pooling_aucpr = sk_m.average_precision_score(unfolded_true_label[~np.isnan(unfolded_true_label)].astype(bool),unfolded_pred_prob[~np.isnan(unfolded_true_label)])
	averaging_aucpr = aucpr_score/n_repfolds
	averaging_sample_variance_aucpr = (aucpr_score2-aucpr_score**2/n_repfolds)/(n_repfolds-1)

	critical_pvalue=0.05
	c = sc_st.t.ppf(1-critical_pvalue/2, df= n_repfolds-1)

	if(n_folds>1):
		std_error_aucroc = np.sqrt(averaging_sample_variance_aucroc*(1/n_repfolds+1/(n_folds-1)))
		std_error_aucpr = np.sqrt(averaging_sample_variance_aucpr*(1/n_repfolds+1/(n_folds-1)))
	else:
		m = (true_label==0).sum()
		n = (true_label==1).sum()
		auc = pooling_aucroc
		pxxy = auc/(2-auc)
		pxyy = 2*auc**2/(1+auc)
		variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
		std_error_aucroc = np.sqrt(variance)
		c=1
		auc = pooling_aucpr
		pxxy = auc/(2-auc)
		pxyy = 2*auc**2/(1+auc)
		variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)
		std_error_aucpr = np.sqrt(variance)

	print('Pooling AUC ROC:', pooling_aucroc)
	print('Averaging AUC ROC:', averaging_aucroc)
	print('Averaging Std Error:', std_error_aucroc)
	print('95% Confidence Interval: [', averaging_aucroc - c*std_error_aucroc,",", averaging_aucroc+c*std_error_aucroc, "]")
	print('Pooling PR ROC:', pooling_aucpr)
	print('Averaging PR ROC:', averaging_aucpr)
	print('Averaging Std Error:', std_error_aucpr)
	print('95% Confidence Interval: [', averaging_aucpr - c*std_error_aucpr,",", averaging_aucpr+c*std_error_aucpr, "]")

	results_dict = {"pool_aucroc": pooling_aucroc,
					"avg_aucroc": averaging_aucroc,
					"avg_aucroc_stderr": std_error_aucroc,
					"aucroc_95ci_low": averaging_aucroc - c*std_error_aucroc,
					"aucroc_95ci_high": averaging_aucroc+c*std_error_aucroc,
					"pool_aucpr": pooling_aucpr,
					"avg_aucpr": averaging_aucpr,
					"avg_aucpr_stderr": std_error_aucpr,
					"aucpr_95ci_low": averaging_aucpr - c*std_error_aucpr,
					"aucpr_95ci_high": averaging_aucpr+c*std_error_aucpr}

	return (unfolded_pred_prob,unfolded_true_label, results_dict)

def paired_ttest(req_name1, req_name2, xlsname, tmp_folder):

	n_reps=0
	n_repfolds=0
	score=0
	score2=0
	for input_file_1, input_file_2 in zip(req_name1.input(), req_name2.input()):
		with open(input_file_1.path, 'rb') as f1:
			with open(input_file_2.path, 'rb') as f2:
				n_reps+=1
				rp_dict1=pickle.load(f1)
				rp_dict2=pickle.load(f2)

				for fold in range(len(rp_dict1['pred_prob'])):
					n_repfolds+=1
					true_label1 = np.array(rp_dict1['true_label'][fold])
					pred_prob1 = np.array(rp_dict1['pred_prob'][fold])
					tl1 = true_label1[~np.isnan(true_label1)]
					pp1 = pred_prob1[~np.isnan(true_label1)]
					auc1 = sk_m.roc_auc_score(tl1,pp1)

					#True labels for the same workflow should be the same and there is no need to load the ones from rp_dict2
					pred_prob2 = np.array(rp_dict2['pred_prob'][fold])
					pp2 = pred_prob2[~np.isnan(true_label1)]
					auc2 = sk_m.roc_auc_score(tl1,pp2)

					score+= auc1-auc2
					score2+=(auc1-auc2)**2

	n_folds=n_repfolds/n_reps

	averaging_diff = score/n_repfolds
	averaging_sample_variance = (score2-score**2/n_repfolds)/(n_repfolds-1)
	if(n_folds>1):
		std_error = np.sqrt(averaging_sample_variance*(1/n_repfolds+1/(n_folds-1)))
	else:
		std_error = 1e100

	t_statistic = averaging_diff/std_error


	pvalue = sc_st.t.sf(np.absolute(t_statistic), df= n_repfolds-1)

	return (averaging_diff, pvalue)

def AUC_stderr_hanley(df, label_name, feature_oddratio):
	m = df.loc[df[label_name]==0,label_name].count()
	n = df.loc[df[label_name]==1,label_name].count()
	Y_prob = pd.Series(0, index=df.index)
	for feat in feature_oddratio.keys():
		Y_prob += feature_oddratio[feat]*df.loc[:,feat]

	auc = sk_m.roc_auc_score(df[label_name],Y_prob)

	pxxy = auc/(2-auc)
	pxyy = 2*auc**2/(1+auc)
	variance = (auc*(1-auc)+(m-1)*(pxxy-auc**2)+(n-1)*(pxyy-auc**2))/(m*n)

	stderr = np.sqrt(variance)

	return (auc, stderr)


def AUC_stderr_classic(df, label_name, feature_oddratio):

	m = df.loc[df[label_name]==0,label_name].count()
	n = df.loc[df[label_name]==1,label_name].count()

	Y_prob = pd.Series(0, index=df.index)
	for feat in feature_oddratio.keys():
		Y_prob += feature_oddratio[feat]*df.loc[:,feat]

	auc = sk_m.roc_auc_score(df[label_name],Y_prob)

	variance = auc*(1-auc)/min(m,n)
	stderr = np.sqrt(variance)

	return (auc, stderr)

def AUC_var_Cortes_Mohri(m,n,k):
	#¿Qué k usamos para este cálculo?
	Z1_num = 0
	Z2_num = 0
	Z3_num = 0
	Z4_num = 0
	Z_den = 0
	for x in range(0, k):
		Z1_num += sc_ms.comb(m+n,x)
	for x in range(0, k-2):
		Z2_num += sc_ms.comb(m+n-2,x)
	for x in range(0, k-2):
		Z3_num += sc_ms.comb(m+n-2,x)
	for x in range(0, k-3):
		Z4_num += sc_ms.comb(m+n-3,x)
	for x in range(0,k+1):
		Z_den+= sc_ms.comb(m+n+1,x)

	Z1=Z1_num/Z_den
	Z2=Z2_num/Z_den
	Z3=Z3_num/Z_den
	Z4=Z4_num/Z_den


	for x in range(0, k-3):
		Z1_num += sc_ms.comb(m+n-3,x)
	for x in range(0,k+1):
		Z1_den+= sc_ms.comb(m+n+1,x)
	Z4=Z4_num/Z4_den

	T = 3*((m-n)**2 + m + n) + 2

	Q0 = (m + n +1)*T*k**2+((-3*n**2+3*m*n+3*m+1)*T-12*(3*m*n+m+n)-8)*k+(-3*m**2+7*m+10*n+3*n*m+10)*T-4*(3*m*n+m+n+1)

	Q1 = T*k**3+3*(m-1)*T*k**2+((-3*n**2+3*m*n-3*m+8)*T-6*(6*m*n+m+n))*k+(-3*m**2+7*(m+n)+3*m*n)*T-2*(6*m*n+m+n)

	variance = (m+n+1)*(m+n)*(m+n-1)*T*((m+n-2)*Z4-(2*m-n+3*k-10)*Z3)/(72*m**2*n**2)+(m+n+1)*(m+n)*T*(m**2-n*m+3*k*m-5*m+2*k**2-n*k+12-9*k)*Z2/(48*m**2*n**2)-(m+n+1)**2*(m-n)**4*Z1**2/(16*m**2*n**2)-(m+n+1)*Q1*Z1/(72*m**2*n**2)+k*Q0/(144*m**2*n**2)

	return variance

def mdaeli5_analysis(data, label, features,clf,clf_name):

	X = data.loc[:,features]
	Y = data.loc[:,[label]]

	eli5_pi = eli5.sklearn.PermutationImportance(clf, scoring='roc_auc', n_iter=20, random_state=None, cv=10)
	eli5_pi.fit(X.values,Y.values)
	pi_cv = eli5_pi.feature_importances_
	std_pi_cv = eli5_pi.feature_importances_std_

	print(clf_name, "eli5 MDA importances with cv=10")
	for value, std, column in sorted(zip(pi_cv, std_pi_cv, X.columns), reverse = True):
		print(f'{column:20}', f'{value/pi_cv.max():4.3f}', f'{std/pi_cv.max():4.3f}')

	# eli5_pi = eli5.sklearn.PermutationImportance(clf, scoring='roc_auc', n_iter=20, random_state=None, cv=None)
	# eli5_pi.fit(X.values,Y.values)
	# pi_cv = eli5_pi.feature_importances_
	# std_pi_cv = eli5_pi.feature_importances_std_
	#
	# print(clf_name, "eli5 MDA importances trained and tested on the same dataset")
	# for value, std, column in sorted(zip(pi_cv, std_pi_cv, X.columns), reverse = True):
	# 	print(f'{column:20}', f'{value/pi_cv.max():4.3f}', f'{std/pi_cv.max():4.3f}')

	return (pi_cv, std_pi_cv)

def mdaeli5_analysis_ext(data, label, features, final_model,clf_name):

	X = data.loc[:,features]
	Y = data.loc[:,[label]]

	eli5_pi = eli5.sklearn.PermutationImportance(final_model, scoring='roc_auc', n_iter=20, random_state=None, cv='prefit')
	eli5_pi.fit(X.values,Y.values)
	pi_cv = eli5_pi.feature_importances_
	std_pi_cv = eli5_pi.feature_importances_std_

	print(clf_name, "eli5 MDA importances on external Dataset")
	for value, std, column in sorted(zip(pi_cv, std_pi_cv, X.columns), reverse = True):
		print(f'{column:20}', f'{value/pi_cv.max():4.3f}', f'{std/pi_cv.max():4.3f}')


	return (pi_cv, std_pi_cv)
