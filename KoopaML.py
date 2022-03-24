import os
import numpy as np
import pandas as pd
import datetime as dt
import pickle
import sklearn.calibration as sk_cal
import logging
import sys

import luigi
import contextlib

from utils.crossvalidation import *
from utils.analysis import *
from user_data_utils import *
from user_external_data_utils import *
from user_MLmodels_info import ML_info
from user_RiskScores_info import RS_info
from user_Workflow_info import WF_info

# Global variables for path folders
TIMESTRING=dt.datetime.now().strftime("%y%m%d-%H%M%S")
log_path = os.path.abspath("log")
tmp_path = os.path.abspath("intermediate")
model_path = os.path.abspath("models")
report_path = os.path.abspath(f"report-{TIMESTRING}")

def setupLog(name):
	try:
		os.makedirs(log_path)
	except:
		pass
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
		filename=os.path.join(log_path, f'{name}.log'),
		filemode='a'
		)

	stdout_logger = logging.getLogger(f'STDOUT_{name}')
	sl = StreamToLogger(stdout_logger, logging.INFO)
	sys.stdout = sl

	stderr_logger = logging.getLogger(f'STDERR_{name}')
	sl = StreamToLogger(stderr_logger, logging.ERROR)
	sys.stderr = sl

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())

#Luigi Tasks
class LoadDatabase(luigi.Task):
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = load_database()
		df_input.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_input.to_excel(writer, sheet_name='Sheet1')
		writer.save()

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_loaded.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_loaded.xlsx"))}

class CleanDatabase(luigi.Task):
	def requires(self):
		return LoadDatabase()
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = clean_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_clean.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_clean.xlsx"))}

class ProcessDatabase(luigi.Task):

	def requires(self):
		return CleanDatabase()
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = process_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_processed.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_processed.xls"))}

class FillnaDatabase(luigi.Task):
	def requires(self):
		return ProcessDatabase()

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = fillna_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_fillna.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "df_fillna.xls"))}

class FilterPreprocessDatabase(luigi.Task):
	wf_name = luigi.Parameter()

	def requires(self):
		return FillnaDatabase()

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		filter_function = WF_info[self.wf_name]["filter_function"]
		df_filtered = filter_function(df_input)
		df_preprocessed = preprocess_filtered_database(df_filtered, self.wf_name)
		df_preprocessed.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_preprocessed.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"df_filtered_preprocessed_{self.wf_name}.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"df_filtered_preprocessed_{self.wf_name}.xls"))}

class CleanExternalDatabase(luigi.Task):
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = load_external_database()
		df_output = clean_external_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_clean.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_clean.xlsx"))}

class ProcessExternalDatabase(luigi.Task):

	def requires(self):
		return CleanExternalDatabase()
	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = process_external_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_processed.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_processed.xls"))}

class FillnaExternalDatabase(luigi.Task):
	def requires(self):
		return ProcessExternalDatabase()

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_output = fillna_external_database(df_input)
		df_output.to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_fillna.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, "external_df_fillna.xls"))}

class FilterPreprocessExternalDatabase(luigi.Task):
	wf_name = luigi.Parameter()

	def requires(self):
		return FillnaExternalDatabase()

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		filter_function = WF_info[self.wf_name]["filter_function"]
		df_filtered = filter_function(df_input)
		df_preprocessed = preprocess_filtered_external_database(df_filtered, self.wf_name)
		df_preprocessed .to_pickle(self.output()["pickle"].path)
		writer = pd.ExcelWriter(self.output()["xls"].path, engine='xlsxwriter')
		df_preprocessed.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"external_df_filtered_preprocessed_{self.wf_name}.pickle")),
				"xls": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"external_df_filtered_preprocessed_{self.wf_name}.xls"))}

class ExternalValidation(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return {'external_data': FilterPreprocessExternalDatabase(self.wf_name),
				'unfiltered_data': FillnaExternalDatabase(),
				'clf': FinalModelAndHyperparameterResults(wf_name = self.wf_name, clf_name = self.clf_name)
				}

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["external_data"]["pickle"].path)
		features = WF_info[self.wf_name]["feature_list"]
		label = WF_info[self.wf_name]["label_name"]
		with open(self.input()["clf"].path, 'rb') as f:
			clf = pickle.load(f)

		tl_pp_dict = external_validation(df_input, label, features, clf)

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.clf_name}.dict"))

class ExternalValidationRS(luigi.Task):
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return FilterPreprocessExternalDatabase(self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]

		tl_pp_dict = external_validation_RS(df_input, label, feature_oddratio_dict)

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.score_name}.dict"))

class ExternalValidationRefittedRS(luigi.Task):
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return {'df': FilterPreprocessExternalDatabase(wf_name = self.wf_name),
				'feature_oddratio': FinalRefittedRSAndOddratios(wf_name = self.wf_name, score_name=self.score_name)}

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()['df']["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		with open(self.input()["feature_oddratio"]['pickle'].path, 'rb') as f:
			feature_oddratio_dict=pickle.load(f)


		tl_pp_dict = external_validation_RS(df_input, label, feature_oddratio_dict)

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"ExternalValidation_PredProb_{self.wf_name}_{self.score_name}.dict"))

class CalculateKFold(luigi.Task):

	seed = luigi.IntParameter()
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return {'data':FilterPreprocessDatabase(self.wf_name),
				'unfiltered_data': FillnaDatabase()}

	def run(self):
		setupLog(self.__class__.__name__)

		df_input = pd.read_pickle(self.input()["unfiltered_data"]["pickle"].path)
		df_filtered = pd.read_pickle(self.input()["data"]["pickle"].path)
		filter_function = WF_info[self.wf_name]["filter_function"]
		features = WF_info[self.wf_name]["feature_list"]
		label = WF_info[self.wf_name]["label_name"]
		group_label = WF_info[self.wf_name]["group_label"]
		cv_type = WF_info[self.wf_name]["validation_type"]
		folds = WF_info[self.wf_name]["cv_folds"]
		clf = ML_info[self.clf_name]["clf"]
		calibration = ML_info[self.clf_name]["calibration"]

		if ((cv_type == 'kfold') or (cv_type=='stratifiedkfold')):
			tl_pp_dict = predict_kfold_ML(df_filtered, label, features, cv_type, clf, calibration, self.seed, folds)
		elif ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
			tl_pp_dict = predict_groupkfold_ML(df_filtered, label, features, group_label, cv_type, clf, calibration,self.seed, folds)
		elif (cv_type == 'unfilteredkfold'):
			tl_pp_dict = predict_filter_kfold_ML(df_input, label, features, filter_function, clf, calibration,self.seed, folds)
		else:
			raise('cv_type not recognized')

		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name,self.clf_name,f"TrueLabel_PredProb_{self.seed}.dict"))

class RiskScore_KFold(luigi.Task):

	seed = luigi.IntParameter()
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return {'data':FilterPreprocessDatabase(self.wf_name),
				'unfiltered_data': FillnaDatabase()}

	def run(self):
		setupLog(self.__class__.__name__)

		df_input = pd.read_pickle(self.input()["unfiltered_data"]["pickle"].path)
		df_filtered = pd.read_pickle(self.input()["data"]["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		filter_function = WF_info[self.wf_name]["filter_function"]
		features = WF_info[self.wf_name]["feature_list"]
		group_label = WF_info[self.wf_name]["group_label"]
		cv_type = WF_info[self.wf_name]["validation_type"]
		folds = WF_info[self.wf_name]["cv_folds"]
		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]

		if ((cv_type == 'kfold') or (cv_type=='stratifiedkfold')):
			tl_pp_dict = predict_kfold_RS(df_filtered, label, features, cv_type, feature_oddratio_dict, self.seed, folds)
		elif ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
			tl_pp_dict = predict_groupkfold_RS(df_filtered, label, features, group_label, cv_type, feature_oddratio_dict,  self.seed, folds)
		elif (cv_type == 'unfilteredkfold'):
			tl_pp_dict = predict_filter_kfold_RS(df_input, label, features, filter_function, feature_oddratio_dict,  self.seed, folds)
		else:
			raise('cv_type not recognized')


		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__, self.wf_name, self.score_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, self.wf_name, self.score_name, f"TrueLabel_PredProb_{self.seed}.dict"))

class RefittedRiskScore_KFold(luigi.Task):
	seed = luigi.IntParameter()
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return {'data':FilterPreprocessDatabase(self.wf_name),
				'unfiltered_data': FillnaDatabase()}

	def run(self):
		setupLog(self.__class__.__name__)

		df_input = pd.read_pickle(self.input()["unfiltered_data"]["pickle"].path)
		df_filtered = pd.read_pickle(self.input()["data"]["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		filter_function = WF_info[self.wf_name]["filter_function"]
		features = WF_info[self.wf_name]["feature_list"]
		group_label = WF_info[self.wf_name]["group_label"]
		cv_type = WF_info[self.wf_name]["validation_type"]
		folds = WF_info[self.wf_name]["cv_folds"]
		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]

		if ((cv_type == 'kfold') or (cv_type=='stratifiedkfold')):
			tl_pp_dict = predict_kfold_refitted_RS(df_filtered, label, features, feature_oddratio_dict,  self.seed, folds)
		elif ((cv_type == 'groupkfold') or (cv_type=='stratifiedgroupkfold')):
			tl_pp_dict = predict_groupkfold_refitted_RS(df_filtered, label, features, group_label, cv_type, feature_oddratio_dict, self.seed, folds)
		elif (cv_type == 'unfilteredkfold'):
			tl_pp_dict = predict_filter_kfold_refitted_RS(df_input, label, features, filter_function, feature_oddratio_dict,  self.seed, folds)
		else:
			raise('cv_type not recognized')


		with open(self.output().path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(tl_pp_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__, self.wf_name, f'REFITTED_{self.score_name}'))
		except:
			pass
		return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, self.wf_name, f'REFITTED_{self.score_name}', f"TrueLabel_PredProb_{self.wf_name}_REFITTED_{self.score_name}_{self.seed}.dict"))


class Evaluate_ML(luigi.Task):

	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if self.ext_val == 'Yes':
			yield ExternalValidation(wf_name=self.wf_name,clf_name=self.clf_name)
		else:
			for i in range(1,WF_info[self.wf_name]['cv_repetitions']+1):
				yield CalculateKFold(wf_name=self.wf_name, seed=i,clf_name=self.clf_name)

	def run(self):
		setupLog(self.__class__.__name__)

		(unfolded_pred_prob,unfolded_true_label, results_dict) = group_files_analyze(self.input(), self.clf_name)
		with open(self.output()["pred_prob"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(unfolded_pred_prob, f, pickle.HIGHEST_PROTOCOL)
		with open(self.output()["true_label"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(unfolded_true_label, f, pickle.HIGHEST_PROTOCOL)
		with open(self.output()["auc_results"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)


	def output(self):
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name))
		except:
			pass

		return {"pred_prob": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"Unfolded_Pred_Prob_{prefix}{self.clf_name}.pickle")),
				"true_label": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"Unfolded_True_Label_{prefix}{self.clf_name}.pickle")),
				"auc_results": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,prefix+self.wf_name,f"AUC_results_{prefix}{self.clf_name}.pickle"))}

class EvaluateRiskScore(luigi.Task):
	wf_name = luigi.Parameter()
	score_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if self.ext_val == 'Yes':
			if(RS_info[self.score_name]['refit_oddratios'] == 'No'):
				yield ExternalValidationRS(wf_name=self.wf_name,score_name=self.score_name)
			elif(RS_info[self.score_name]['refit_oddratios'] == 'Yes'):
				yield ExternalValidationRefittedRS(wf_name=self.wf_name, score_name=self.score_name)
			else:
				raise(f"invalid 'refit_or' value in score {score_name}")
		else:
			for i in range(1,WF_info[self.wf_name]['cv_repetitions']+1):
				if(RS_info[self.score_name]['refit_oddratios'] == 'No'):
					yield RiskScore_KFold(wf_name=self.wf_name, seed=i,score_name=self.score_name)
				elif(RS_info[self.score_name]['refit_oddratios'] == 'Yes'):
					yield RefittedRiskScore_KFold(wf_name=self.wf_name, seed=i,score_name=self.score_name)
				else:
					raise(f"invalid 'refit_or' value in score {score_name}")

	def run(self):
		setupLog(self.__class__.__name__)

		(unfolded_pred_prob,unfolded_true_label,results_dict) = group_files_analyze(self.input(), self.score_name)
		with open(self.output()["pred_prob"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(unfolded_pred_prob, f, pickle.HIGHEST_PROTOCOL)
		with open(self.output()["true_label"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(unfolded_true_label, f, pickle.HIGHEST_PROTOCOL)
		with open(self.output()["auc_results"].path, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

	def output(self):
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__,prefix+self.wf_name))
		except:
			pass
		return {"pred_prob": luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__,prefix+self.wf_name,f"Unfolded_Pred_Prob_{prefix}{self.score_name}.pickle")),
				"true_label": luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__,prefix+self.wf_name,f"Unfolded_True_Label_{prefix}{self.score_name}.pickle")),
				"auc_results": luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__,prefix+self.wf_name,f"AUC_results_{prefix}{self.score_name}.pickle"))}

class ConfidenceIntervalHanleyRS(luigi.Task):
	wf_name = luigi.Parameter()
	score_name = luigi.Parameter()
	ext_val = luigi.Parameter(default = 'No')

	def requires(self):
		if (self.ext_val == 'No'):
			return FilterPreprocessDatabase(self.wf_name)
		elif (self.ext_val == 'Si'):
			return FilterPreprocessExternalDatabase(self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		(auc, stderr) = AUC_stderr_classic(df_input, label_name=WF_info[self.wf_name]["label_name"], feature_oddratio=RS_info[self.score_name]["feature_oddratio"])
		ci95_low= auc-1.96*stderr
		ci95_high= auc+1.96*stderr


		with open(self.output().path,'w') as f:
			f.write(f"Confidence Interval Upper Bound Classic(95%): {auc} ({ci95_low}-{ci95_high})\n")

			(auc, stderr) = AUC_stderr_hanley(df_input , label_name=WF_info[self.wf_name]["label_name"], feature_oddratio=RS_info[self.score_name]["feature_oddratio"])
			ci95_low= auc-1.96*stderr
			ci95_high= auc+1.96*stderr
			f.write(f"Confidence Interval Hanley(95%): {ci95_low}-{ci95_high}\n")
	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path, self.__class__.__name__, self.wf_name))
		except:
			pass
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		return luigi.LocalTarget(os.path.join(tmp_path, self.__class__.__name__, self.wf_name, f"CI_{prefix}{self.score_name}.txt"))


class DescriptiveXLS(luigi.Task):
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if(self.ext_val == 'Yes'):
			return ProcessExternalDatabase()
		else:
			return ProcessDatabase()

	def run(self):
		setupLog(self.__class__.__name__)
		df_input = pd.read_pickle(self.input()["pickle"].path)
		df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
		label = WF_info[self.wf_name]["label_name"]

		df_output=create_descriptive_xls(df_filtered, self.wf_name, label)
		writer = pd.ExcelWriter(self.output().path, engine='xlsxwriter')
		df_output.to_excel(writer, sheet_name='Sheet1')
		writer.save()


	def output(self):
		try:
			os.makedirs(os.path.join(report_path, self.wf_name))
		except:
			pass
		if(self.ext_val == 'No'):
			return luigi.LocalTarget(os.path.join(report_path, self.wf_name, f"{self.wf_name}_descriptivo.xlsx"))
		elif(self.ext_val == 'Yes'):
			return luigi.LocalTarget(os.path.join(report_path, self.wf_name, f"{self.wf_name}_descriptivo_EXT.xlsx"))

class FinalModelAndHyperparameterResults(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return FilterPreprocessDatabase(self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_filtered = pd.read_pickle(self.input()["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		features = WF_info[self.wf_name]["feature_list"]
		group_label = WF_info[self.wf_name]["group_label"]

		self.clf=ML_info[self.clf_name]["clf"]
		calibration = ML_info[self.clf_name]["calibration"]

		X_full = df_filtered.loc[:,features]
		Y_full = df_filtered.loc[:,[label]]

		X = X_full.loc[~Y_full[label].isnull()]
		Y = Y_full.loc[~Y_full[label].isnull()]

		if(group_label is None):
			self.clf.fit(X,Y.values.ravel().astype(int))
		else:
			G_full = df_filtered.loc[:,[group_label]]
			G = G_full.loc[~Y_full[label].isnull()]
			try:
				self.clf.fit(X,Y.values.ravel().astype(int),groups=G)
			except:
				self.clf.fit(X,Y.values.ravel().astype(int))

		if hasattr(self.clf, 'best_estimator_'):
			writer = pd.ExcelWriter(os.path.join(model_path,self.wf_name,f"HyperparameterResults_{self.wf_name}_{self.clf_name}.xlsx"), engine='xlsxwriter')
			pd.DataFrame(self.clf.cv_results_).to_excel(writer, sheet_name='Sheet1')
			writer.save()

		if(calibration is None):
			self.calibrated_clf = self.clf
		elif(calibration == 'isotonic'):
			if hasattr(clf, 'best_estimator_'):
				self.calibrated_clf = sk_cal.CalibratedClassifierCV(self.clf.best_estimator_, method='isotonic', cv=10)
			else:
				self.calibrated_clf = sk_cal.CalibratedClassifierCV(self.clf, method='isotonic', cv=10)
			try:
				self.calibrated_clf.fit(X,Y.values.ravel().astype(int),groups=G)
			except:
				self.calibrated_clf.fit(X,Y.values.ravel().astype(int))
		elif(calibration == 'sigmoid'):
			if hasattr(self.clf, 'best_estimator_'):
				self.calibrated_clf = sk_cal.CalibratedClassifierCV(self.clf.best_estimator_, method='sigmoid', cv=10)
			else:
				self.calibrated_clf = sk_cal.CalibratedClassifierCV(self.clf, method='sigmoid', cv=10)
			try:
				self.calibrated_clf.fit(X,Y.values.ravel().astype(int),groups=G)
			except:
				self.calibrated_clf.fit(X,Y.values.ravel().astype(int))
		else:
			print('unknown calibration type')
			raise
		with open(self.output().path,'wb') as f:
			pickle.dump(self.calibrated_clf, f, pickle.HIGHEST_PROTOCOL)


	def output(self):
		try:
			os.makedirs(os.path.join(model_path,self.wf_name, self.clf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(model_path,self.wf_name,self.clf_name,f"ML_model_{self.wf_name}_{self.clf_name}.pickle"))

class FinalRefittedRSAndOddratios(luigi.Task):
	#jgkñjfsdñajfasñdjfñasjkfdasjdfañksdfjañsdjfkafsñ
	score_name = luigi.Parameter()
	wf_name = luigi.Parameter()

	def requires(self):
		return FilterPreprocessDatabase(self.wf_name)

	def run(self):
		setupLog(self.__class__.__name__)
		df_filtered = pd.read_pickle(self.input()["pickle"].path)
		label = WF_info[self.wf_name]["label_name"]
		features = WF_info[self.wf_name]["feature_list"]
		feature_oddratio_dict = RS_info[self.score_name]["feature_oddratio"]

		refitted_or =  refitted_oddratios(df_filtered, label, feature_oddratio_dict)

		with open(self.output()['pickle'].path,'wb') as f:
			pickle.dump(refitted_or, f, pickle.HIGHEST_PROTOCOL)

		with open(self.output()['txt'].path,'w') as f:
			for feat in refitted_or.keys():
				f.write(f'{feat}: {refitted_or[feat]}\n')

	def output(self):
		try:
			os.makedirs(os.path.join(model_path,self.wf_name,self.score_name))
		except:
			pass
		return {'pickle': luigi.LocalTarget(os.path.join(model_path,self.wf_name,self.score_name,f"Refitted_Oddratios_{self.wf_name}_{self.score_name}.pickle")),
				'txt': luigi.LocalTarget(os.path.join(model_path,self.wf_name,self.score_name,f"Refitted_Oddratios_{self.wf_name}_{self.score_name}.txt"))}



class AllModels_PairedTTest(luigi.Task):
	wf_name = luigi.Parameter()
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	ext_val = luigi.Parameter(default = 'No')

	def requires(self):
		requirements={}
		for clf_or_score1 in self.list_RS:
			requirements[clf_or_score1] = EvaluateRiskScore(wf_name=self.wf_name, score_name = clf_or_score1, ext_val=self.ext_val)
		for clf_or_score2 in self.list_ML:
			requirements[clf_or_score2] = Evaluate_ML(wf_name=self.wf_name, clf_name=clf_or_score2, ext_val=self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			list_comparisons = []
			for clf_or_score1 in self.list_ML+self.list_RS:
				for clf_or_score2 in self.list_ML+self.list_RS:
					if (clf_or_score1 != clf_or_score2):
						(averaging_diff, pvalue) = paired_ttest(self.requires()[clf_or_score1],self.requires()[clf_or_score2], self.wf_name, tmp_path)
						if clf_or_score1 in self.list_ML:
							formal_name1 = ML_info[clf_or_score1]["formal_name"]
						else:
							formal_name1 = RS_info[clf_or_score1]["formal_name"]
						if clf_or_score2 in self.list_ML:
							formal_name2 = ML_info[clf_or_score2]["formal_name"]
						else:
							formal_name2 = RS_info[clf_or_score2]["formal_name"]
						wf_formal_title = WF_info[self.wf_name]["formal_title"]
						f.write(f"{wf_formal_title}: {formal_name1}-{formal_name2}, Avg Diff: {averaging_diff}, p-value: {pvalue}\n")

	def output(self):
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		try:
			os.makedirs(os.path.join(report_path, self.wf_name))
		except:
			pass
		return luigi.LocalTarget(os.path.join(report_path, self.wf_name,f"AllModelsPairedTTest_{prefix}{self.wf_name}.txt"))

class GraphsWF(luigi.Task):

	wf_name = luigi.Parameter()
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	n_best_ML = luigi.IntParameter(default=1)
	n_best_RS = luigi.IntParameter(default=4)
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		requirements = {}
		for i in self.list_ML:
			requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		for i in self.list_RS:
			requirements[i] = EvaluateRiskScore(score_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)

		# First we plot every ML model and risk score
		plot_all_rocs(task_requires=self.input(), fig_path= self.output()["roc_all"].path,title=WF_info[self.wf_name]["formal_title"])
		plot_all_prs(task_requires=self.input(), fig_path= self.output()["pr_all"].path,title=WF_info[self.wf_name]["formal_title"])

		# Second we open the results dictionary for every ML model and Risk Score in the workflow wf_name to determine
		# the best ML model and the best risk score
		if((len(self.list_RS) > 0) & (len(self.list_ML) > 0)):
			if(self.n_best_ML > len(self.list_ML)):
				self.n_best_ML = len(self.list_ML)
			if(self.n_best_RS > len(self.list_RS)):
				self.n_best_RS = len(self.list_RS)
			auc_ml = {}
			for i in self.list_ML:
				with open(self.input()[i]["auc_results"].path, 'rb') as f:
					results_dict=pickle.load(f)
					auc_ml[i]=results_dict["avg_aucroc"]

			sorted_ml = sorted(auc_ml.keys(), key=lambda x: auc_ml[x], reverse=True)

			auc_rs = {}
			for i in self.list_RS:
				with open(self.input()[i]["auc_results"].path, 'rb') as f:
						results_dict=pickle.load(f)
						auc_rs[i]=results_dict["avg_aucroc"]

			sorted_rs = sorted(auc_rs.keys(), key=lambda x: auc_rs[x], reverse=True)

			# We plot the ROC of the best ML model and the best risk score
			tasks={}
			for ml in sorted_ml[0:self.n_best_ML]:
				tasks[ml]=self.input()[ml]
			for rs in sorted_rs[0:self.n_best_RS]:
				tasks[rs]=self.input()[rs]

			plot_all_rocs(task_requires=tasks , fig_path= self.output()["roc_best"].path,title=WF_info[self.wf_name]["formal_title"])
			plot_all_prs(task_requires=tasks , fig_path= self.output()["pr_best"].path,title=WF_info[self.wf_name]["formal_title"])

	def output(self):
		try:
			os.makedirs(os.path.join(report_path,self.wf_name))
		except:
			pass
		if(self.ext_val == 'Yes'):
			prefix = 'EXT_'
		else:
			prefix = ''
		if((len(self.list_RS) > 0) & (len(self.list_ML) > 0)):
			return {"roc_all": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"AllModelsROC_{prefix}{self.wf_name}.png")),
					"roc_best": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"BestModelsROC_{prefix}{self.wf_name}.png")),
					"pr_all": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"AllModelsPR_{prefix}{self.wf_name}.png")),
					"pr_best": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"BestModelsPR_{prefix}{self.wf_name}.png"))}
		else:
			return {"roc_all": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"AllModelsROC_{prefix}{self.wf_name}.png")),
					"pr_all": luigi.LocalTarget(os.path.join(report_path,self.wf_name, f"AllModelsPR_{prefix}{self.wf_name}.png"))}



class ThresholdPoints(luigi.Task):
	clf_or_score=luigi.Parameter()
	wf_name = luigi.Parameter()
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	ext_val = luigi.Parameter(default = 'No')


	def requires(self):
		if (self.clf_or_score in self.list_RS):
			return EvaluateRiskScore(wf_name=self.wf_name, score_name = self.clf_or_score, ext_val = self.ext_val)
		elif (self.clf_or_score in self.list_ML):
			return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, ext_val=self.ext_val)
		else:
			raise Exception(f"{self.clf_score} not in list_ristkscores or list_MLmodels")

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.input()["pred_prob"].path, 'rb') as f:
			pred_prob=pickle.load(f)
		with open(self.input()["true_label"].path, 'rb') as f:
			true_label=pickle.load(f)

		with open(self.output().path,'w') as f:
			(best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_accuracy(pred_prob, true_label)
			f.write(f'Threshold: {best_threshold} Optimum for accuracy\n')
			f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
			f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
			f.write("\n")

			(best_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_single(pred_prob, true_label)
			f.write(f'Threshold: {best_threshold} Optimum for single point AUC\n')
			f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
			f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
			f.write("\n")

			threshold_dict = cutoff_threshold_double(pred_prob, true_label)
			(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = threshold_dict["threshold1"]
			(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = threshold_dict["threshold2"]

			f.write('Optimum for double point AUC\n')
			f.write(f'Threshold: {best_threshold1}\n')
			f.write(f'TP:{tprate1*100:.1f} FP:{fprate1*100:.1f} TN:{tnrate1*100:.1f} FN:{fnrate1*100:.1f}\n')
			f.write(f'Sensitivity:{sens1*100:.1f} Specifity:{spec1*100:.1f} Precision:{prec1*100:.1f} NPRv:{nprv1*100:.1f}\n')
			f.write(f'Threshold: {best_threshold2}\n')
			f.write(f'TP:{tprate2*100:.1f} FP:{fprate2*100:.1f} TN:{tnrate2*100:.1f} FN:{fnrate2*100:.1f}\n')
			f.write(f'Sensitivity:{sens2*100:.1f} Specifity:{spec2*100:.1f} Precision:{prec2*100:.1f} NPRv:{nprv2*100:.1f}\n')
			f.write("\n")

			threshold_dict = cutoff_threshold_triple(pred_prob, true_label)
			(best_threshold1, tprate1, fprate1, tnrate1, fnrate1, sens1, spec1, prec1, nprv1) = threshold_dict["threshold1"]
			(best_threshold2, tprate2, fprate2, tnrate2, fnrate2, sens2, spec2, prec2, nprv2) = threshold_dict["threshold2"]
			(best_threshold3, tprate3, fprate3, tnrate3, fnrate3, sens3, spec3, prec3, nprv3) = threshold_dict["threshold3"]

			f.write('Optimum for triple point AUC\n')
			f.write(f'Threshold: {best_threshold1}\n')
			f.write(f'TP:{tprate1*100:.1f} FP:{fprate1*100:.1f} TN:{tnrate1*100:.1f} FN:{fnrate1*100:.1f}\n')
			f.write(f'Sensitivity:{sens1*100:.1f} Specifity:{spec1*100:.1f} Precision:{prec1*100:.1f} NPRv:{nprv1*100:.1f}\n')
			f.write(f'Threshold: {best_threshold2}\n')
			f.write(f'TP:{tprate2*100:.1f} FP:{fprate2*100:.1f} TN:{tnrate2*100:.1f} FN:{fnrate2*100:.1f}\n')
			f.write(f'Sensitivity:{sens2*100:.1f} Specifity:{spec2*100:.1f} Precision:{prec2*100:.1f} NPRv:{nprv2*100:.1f}\n')
			f.write(f'Threshold: {best_threshold3}\n')
			f.write(f'TP:{tprate3*100:.1f} FP:{fprate3*100:.1f} TN:{tnrate3*100:.1f} FN:{fnrate3*100:.1f}\n')
			f.write(f'Sensitivity:{sens3*100:.1f} Specifity:{spec3*100:.1f} Precision:{prec3*100:.1f} NPRv:{nprv3*100:.1f}\n')
			f.write("\n")

			for beta in [0.5,1,2]:
				(max_f1_threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = cutoff_threshold_maxfbeta(pred_prob, true_label, beta)
				f.write(f'Threshold: {max_f1_threshold} Optimum for f{beta}\n')
				f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
				f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
				f.write("\n")

	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		if self.ext_val == 'No':
			return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.txt"))
		elif self.ext_val == 'Yes':
			return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.txt"))

class BestMLModelReport(luigi.Task):
	wf_name = luigi.Parameter()
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	ext_val = luigi.Parameter(default='No')
	all_ML_importances = luigi.BoolParameter(default=True)

	def requires(self):
		requirements = {}
		for i in self.list_ML:
			requirements[i] = Evaluate_ML(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
			requirements[i+'_threshold'] = ThresholdPoints(clf_or_score = i, wf_name = self.wf_name, list_ML = self.list_ML, ext_val=self.ext_val)
			if self.all_ML_importances:
				requirements[i+'_importances'] = MDAFeatureImportances(clf_name = i, wf_name = self.wf_name, ext_val=self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)
		# First we open the results dictionary for every ML model in the workflow wf_name to determine
		# the best ML model
		auc_ml = {}
		for i in self.list_ML:
			with open(self.input()[i]["auc_results"].path, 'rb') as f:
				results_dict=pickle.load(f)
				auc_ml[i]=results_dict["avg_aucroc"]

		best_ml = max(auc_ml.keys(), key=(lambda k: auc_ml[k]))

		with open(self.input()[best_ml]["auc_results"].path, 'rb') as f:
			best_ml_results_dict=pickle.load(f)

		with open(self.output().path,'w') as f:
			f.write(f"Model name: {best_ml}\n")
			f.write(f"AUC ROC: {best_ml_results_dict['avg_aucroc']}\n")
			f.write(f"AUC ROC stderr: {best_ml_results_dict['avg_aucroc_stderr']}\n")
			f.write(f"AUC ROC Confidence Interval (95%): {best_ml_results_dict['aucroc_95ci_low']}-{best_ml_results_dict['aucroc_95ci_high']}\n")
			f.write(f"AUC PR: {best_ml_results_dict['avg_aucpr']}\n")
			f.write(f"AUC PR stderr: {best_ml_results_dict['avg_aucpr_stderr']}\n")
			f.write(f"AUC PR Confidence Interval (95%): {best_ml_results_dict['aucpr_95ci_low']}-{best_ml_results_dict['aucpr_95ci_high']}\n")
			f.write("\n")
			with open(self.input()[best_ml+'_threshold'].path, 'r') as f2:
				for line in f2.readlines():
					f.write(line)

			if self.all_ML_importances:
				with open(self.input()[best_ml+'_importances'].path, 'r') as f3:
					for line in f3.readlines():
						f.write(line)
			else:
				prerequisite = MDAFeatureImportances(clf_name = best_ml, wf_name = self.wf_name, ext_val = self.ext_val)
				luigi.build([prerequisite], local_scheduler = False)
				with open(prerequisite.output().path, 'r') as f3:
					for line in f3.readlines():
						f.write(line)

	def output(self):
		try:
			os.makedirs(os.path.join(report_path,self.wf_name))
		except:
			pass
		if self.ext_val == 'No':
			return luigi.LocalTarget(os.path.join(report_path,self.wf_name,f"BestML_Model_report_{self.wf_name}.txt"))
		elif self.ext_val == 'Yes':
			return luigi.LocalTarget(os.path.join(report_path,self.wf_name,f"BestML_Model_report_{self.wf_name}_EXT.txt"))

class BestRSReport(luigi.Task):
	wf_name = luigi.Parameter()
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	ext_val = luigi.Parameter(default = 'No')

	def requires(self):
		requirements = {}
		for rs in self.list_RS:
			requirements[rs] = EvaluateRiskScore(score_name = rs, wf_name = self.wf_name, ext_val = self.ext_val)
			requirements[rs+'_threshold'] = AllThresholds(clf_or_score = rs, wf_name = self.wf_name, list_RS=self.list_RS, ext_val = self.ext_val)
			if (RS_info[rs]['refit_oddratios']=='No'):
				requirements[rs+'_hanley'] = ConfidenceIntervalHanleyRS(score_name = rs, wf_name = self.wf_name, ext_val = self.ext_val)
		return requirements

	def run(self):
		setupLog(self.__class__.__name__)
		# First we open the results dictionary for every ML model in the workflow wf_name to determine
		# the best risk score
		auc_rs = {}
		for i in self.list_RS:
			with open(self.input()[i]["auc_results"].path, 'rb') as f:
				results_dict=pickle.load(f)
				auc_rs[i]=results_dict["avg_aucroc"]

		best_rs = max(auc_rs.keys(), key=(lambda k: auc_rs[k]))

		with open(self.input()[best_rs]["auc_results"].path, 'rb') as f:
			best_rs_results_dict=pickle.load(f)

		with open(self.output().path,'w') as f:
			f.write(f"Score name: {RS_info[best_rs]['formal_name']}\n")
			f.write(f"AUC ROC: {best_rs_results_dict['avg_aucroc']}\n")
			f.write(f"AUC ROC stderr: {best_rs_results_dict['avg_aucroc_stderr']}\n")
			f.write(f"AUC ROC Confidence Interval Subsampling(95%): {best_rs_results_dict['aucroc_95ci_low']}- {best_rs_results_dict['aucroc_95ci_high']}\n")
			if (RS_info[best_rs]['refit_oddratios'] == 'No'):
				with open(self.input()[best_rs+'_hanley'].path, 'r') as f2:
					for line in f2.readlines():
						f.write(line)
			f.write("\n")
			f.write(f"AUC PR: {best_rs_results_dict['avg_aucpr']}\n")
			f.write(f"AUC PR stderr: {best_rs_results_dict['avg_aucpr_stderr']}\n")
			f.write(f"AUC PR Confidence Interval Subsampling(95%): {best_rs_results_dict['aucpr_95ci_low']}- {best_rs_results_dict['aucpr_95ci_high']}\n")
			with open(self.input()[best_rs+'_threshold']['txt'].path, 'r') as f3:
				for line in f3.readlines():
					f.write(line)
	def output(self):
		try:
			os.makedirs(os.path.join(report_path,self.wf_name))
		except:
			pass
		if self.ext_val == 'No':
			return luigi.LocalTarget(os.path.join(report_path,self.wf_name,f"BestRS_report_{self.wf_name}.txt"))
		elif self.ext_val  == 'Yes':
			return luigi.LocalTarget(os.path.join(report_path,self.wf_name,f"BestRS_report_{self.wf_name}_EXT.txt"))


class MDAFeatureImportances(luigi.Task):
	clf_name = luigi.Parameter()
	wf_name = luigi.Parameter()
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if self.ext_val == 'No':
			return {'df': FilterPreprocessDatabase(self.wf_name)}
		elif self.ext_val == 'Yes':
			return {'df': FilterPreprocessExternalDatabase(self.wf_name),
					'clf': FinalModelAndHyperparameterResults(clf_name=self.clf_name, wf_name=self.wf_name)}

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			with contextlib.redirect_stdout(f):
				df_input = pd.read_pickle(self.input()['df']["pickle"].path)
				df_filtered = WF_info[self.wf_name]["filter_function"](df_input)
				label = WF_info[self.wf_name]["label_name"]
				features = WF_info[self.wf_name]["feature_list"]

				if self.ext_val == 'No':
					(pi_cv, std_pi_cv) = mdaeli5_analysis(df_filtered, label, features, clf=ML_info[self.clf_name]["clf"],clf_name=self.clf_name)
				elif self.ext_val == 'Yes':
					with open(self.input()["clf"].path, 'rb') as f:
						clf = pickle.load(f)
					(pi_cv, std_pi_cv) = mdaeli5_analysis_ext(df_filtered, label, features, final_model = clf,clf_name=self.clf_name)
					pass
	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass

		if self.ext_val == 'No':
			return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"MDAeli5_Log_{self.wf_name}_{self.clf_name}.txt"))
		elif self.ext_val == 'Yes':
			return luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,f"MDAeli5_Log_{self.wf_name}_{self.clf_name}_EXT.txt"))

class AllThresholds(luigi.Task):
	clf_or_score=luigi.Parameter()
	wf_name = luigi.Parameter()
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	ext_val = luigi.Parameter(default='No')

	def requires(self):
		if (self.clf_or_score in self.list_RS):
			return EvaluateRiskScore(wf_name=self.wf_name, score_name = self.clf_or_score, ext_val = self.ext_val)
		elif (self.clf_or_score in self.list_ML):
			return Evaluate_ML(wf_name=self.wf_name, clf_name=self.clf_or_score, ext_val = self.ext_val)
		else:
			raise Exception(f"{self.clf_score} not in list_ristkscores or list_MLmodels")

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.input()["pred_prob"].path, 'rb') as f:
			pred_prob=pickle.load(f)
		with open(self.input()["true_label"].path, 'rb') as f:
			true_label=pickle.load(f)

		list_thresholds = all_thresholds(pred_prob, true_label)

		with open(self.output()['txt'].path,'w') as f:
			rows = []
			for i in list_thresholds:
				(threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv) = i
				f.write(f'Threshold: {threshold}\n')
				f.write(f'TP:{tprate*100:.1f} FP:{fprate*100:.1f} TN:{tnrate*100:.1f} FN:{fnrate*100:.1f}\n')
				f.write(f'Sensitivity:{sens*100:.1f} Specifity:{spec*100:.1f} Precision:{prec*100:.1f} NPRv:{nprv*100:.1f}\n')
				rows.append([threshold, tprate, fprate, tnrate, fnrate, sens, spec, prec, nprv])
		df_thr = pd.DataFrame(rows, columns=['Threshold','TP','FP','TN','FN', 'sensitivity','specificity','precision','nprv'])
		with open(self.output()['df'].path,'w') as f:
			df_thr.to_csv(f)
	def output(self):
		try:
			os.makedirs(os.path.join(tmp_path,self.__class__.__name__))
		except:
			pass
		if self.ext_val == 'No':
			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.txt")),
					'df': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}.csv"))}
		elif self.ext_val == 'Yes':
			return {'txt': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.txt")),
					'df': luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__, f"Thresholds_{self.wf_name}_{self.clf_or_score}_EXT.csv"))}

class OnlyGraphs(luigi.Task):

	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))

	def requires(self):

		for it_wf_name in self.list_WF:
			yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS)


	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		TIMESTRING=dt.datetime.now().strftime("%y%m%d-%H%M%S")
		return luigi.LocalTarget(os.path.join(log_path, f"OnlyGraphs_Log-{TIMESTRING}.txt"))

# class FeatureScorer(luigi.Task):
# 	wf_name = luigi.Parameter()
# 	fs_name = luigi.Parameter()
#
# 	def requires(self):
# 		for i in range(1,WF_info[self.wf_name]['cv_repetitions']+1):
# 			yield FeatureScoringFolds(seed = i, cvfolds = WF_info[self.wf_name]['cvfolds'], wf_name = self.wf_name, fs_name = self.fs_name)
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
# 		pass
#
# 	def output(self):
# 		pass
#
# class FeatureScoringFolds(luigi.Task):
# 	seed = luigi.IntParameter()
# 	cvfolds = luigi.IntParameter()
# 	wf_name = luigi.Parameter()
# 	fs_name = luigi.Parameter()
#
# 	def requires(self):
# 		return FillnaDatabase()
#
# 	def run(self):
# 		setupLog(self.__class__.__name__)
#
# 		df_input = pd.read_pickle(self.input()["pickle"].path)
# 		filter_function = WF_info[self.wf_name]["filter_function"]
# 		df_filtered = filter_function(df_input)
# 		features = WF_info[self.wf_name]["feature_list"]
# 		label = WF_info[self.wf_name]["label_name"]
# 		group_label = WF_info[self.wf_name]["group_label"]
# 		cv_type = WF_info[self.wf_name]["validation_type"]
# 		fs_function = FS_info[self.fs_name]["scorer_function"]
#
# 		X = df_filtered.loc[:, features]
# 		Y = df_filtered.loc[:,[label]].astype(bool)
#
# 		if (cv_type == 'kfold'):
# 			kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
# 		elif(cv_type == 'stratifiedkfold'):
# 			kf = sk_ms.StratifiedKFold(cvfolds, random_state=seed, shuffle=True)
# 		elif(cv_type == 'groupkfold'):
# 			kf = GroupKFold(cvfolds)
# 		elif(cv_type == 'stratifiedgroupkfold'):
# 			kf = StratifiedGroupKFold(cvfolds, random_state=seed, shuffle=True)
# 		elif (cv_type == 'unfilteredkfold'):
# 			kf = sk_ms.KFold(cvfolds, random_state=seed, shuffle=True)
# 		else:
# 			raise('cv_type not recognized')
#
# 		if ((cv_type == 'kfold') or (cv_type == 'stratifiedkfold') or (cv_type == 'unfilteredkfold')):
# 			fold = 0
# 			for train_index, test_index in kf.split(X,Y):
# 				X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# 				Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
#
# 				X_train = X_train[~np.isnan(Y_train)]
# 				Y_train = Y_train[~np.isnan(Y_train)].astype(bool)
#
# 				feature_scores = fs_function(X_train,Y_train)
#
# 				scores_dict = dict(zip(X_train.columns, feature_scores))
# 				with open(self.output()[fold]["pickle"].path, 'wb') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)
#
# 				with open(self.output()[fold]["pickle"].path, 'w') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					for (feature, score) in zip(X_train.columns, feature_scores):
# 						f.write(f"{feature}, {score}\n")
# 				fold+=1
#
# 		if ((cv_type == 'groupkfold') or (cv_type == 'stratifiedgroupkfold')):
# 			G = df_filtered.loc[:,[group_label]]
# 			fold = 1
# 			for train_index, test_index in kf.split(X,Y,G):
# 				X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# 				Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
#
# 				X_train = X_train[~np.isnan(Y_train)]
# 				Y_train = Y_train[~np.isnan(Y_train)].astype(bool)
#
# 				feature_scores = fs_function(X_train,Y_train)
#
# 				scores_dict = dict(zip(X_train.columns, feature_scores))
# 				with open(self.output()[fold]["pickle"].path, 'wb') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)
#
# 				with open(self.output()[fold]["pickle"].path, 'w') as f:
# 					# Pickle the 'data' dictionary using the highest protocol available.
# 					for (feature, score) in zip(X_train.columns, feature_scores):
# 						f.write(f"{feature}, {score}\n")
# 				fold+=1
#
# 		pass
# 	def output(self):
# 		try:
# 			os.makedirs(os.path.join(tmp_path,self.__class__.__name__,self.wf_name))
# 		except:
# 			pass
# 		for i in range(1,cvfolds+1):
# 			yield {"pickle": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name, f"FeatureScores_{self.FS_name}_r{self.seed}_f{i}.pickle")),
# 					"txt": luigi.LocalTarget(os.path.join(tmp_path,self.__class__.__name__,self.wf_name, f"FeatureScores_{self.FS_name}_r{self.seed}_f{i}.txt"))}
#

class AllTasks(luigi.Task):

	list_ML = luigi.ListParameter(default=list(ML_info.keys()))
	list_RS = luigi.ListParameter(default=list(RS_info.keys()))
	list_WF = luigi.ListParameter(default=list(WF_info.keys()))


	def __init__(self, *args, **kwargs):
		super(AllTasks, self).__init__(*args, **kwargs)

	def requires(self):

		for it_wf_name in self.list_WF:
			yield DescriptiveXLS(wf_name = it_wf_name)
			yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS)
			if(len(self.list_ML) > 0):
				yield BestMLModelReport(wf_name = it_wf_name, list_ML=self.list_ML)
				# for it_ml_name in self.list_ml:
					# yield AllThresholds(clf_or_score = it_ml_name, wf_name = it_wf_name, list_ML = self.list_ML, ext_val='No')
			if(len(self.list_RS) > 0):
				yield BestRSReport(wf_name = it_wf_name, list_RS=self.list_RS)
			yield AllModels_PairedTTest(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS)
			for it_clf_name in self.list_ML:
				yield FinalModelAndHyperparameterResults(wf_name = it_wf_name, clf_name = it_clf_name)
			for it_rs_name in self.list_RS:
				if(RS_info[it_rs_name]['refit_oddratios']=='Yes'):
					yield FinalRefittedRSAndOddratios(wf_name = it_wf_name, score_name = it_rs_name)
			if(WF_info[it_wf_name]['external_validation'] == 'Yes'):
				yield DescriptiveXLS(wf_name = it_wf_name, ext_val = 'Yes')
				yield GraphsWF(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS, ext_val = 'Yes')
				if(len(self.list_ML) > 0):
					yield BestMLModelReport(wf_name = it_wf_name, list_ML=self.list_ML, ext_val = 'Yes')
					# for it_ml_name in self.list_ml:
						# yield AllThresholds(clf_or_score = it_ml_name, wf_name = it_wf_name, list_ML = self.list_ML, ext_val='Yes')
				if(len(self.list_RS) > 0):
					yield BestRSReport(wf_name = it_wf_name, list_RS=self.list_RS, ext_val = 'Yes')
				yield AllModels_PairedTTest(wf_name = it_wf_name, list_ML=self.list_ML, list_RS=self.list_RS,ext_val = 'Yes')

	def run(self):
		setupLog(self.__class__.__name__)
		with open(self.output().path,'w') as f:
			f.write("prueba\n")

	def output(self):
		TIMESTRING=dt.datetime.now().strftime("%y%m%d-%H%M%S")
		return luigi.LocalTarget(os.path.join(log_path, f"AllTask_Log-{TIMESTRING}.txt"))
