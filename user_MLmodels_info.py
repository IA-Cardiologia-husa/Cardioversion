# In this archive we have to define the dictionary ml_info. This is a dictionary of dictionaries, that for each of the ML models we want
# assigns a dictionary that contains:
#
# clf: a scikit-learn classifier, or any object that implements the functions fit, and predict_proba or decision_function in the same way.
# formal_name: name to be used in plots and report
#
# In this archive we provide 4 examples:
# RF for Random Forest
# BT for Boosted Trees
# LR for Logistic Regression
# RF_pipeline for a Random Forest with hyperparameter tuning including the choice of feature selection strategy

import sklearn.ensemble as sk_en
import sklearn.linear_model as sk_lm
import sklearn.pipeline as sk_pl
import sklearn.model_selection as sk_ms
import sklearn.preprocessing as sk_pp
import sklearn.discriminant_analysis as sk_da
import xgboost as xgb
from utils.featureselecter import FeatureSelecter


ML_info ={}


pipeline_rf = sk_pl.Pipeline(steps=[("rf",sk_en.RandomForestClassifier(n_estimators = 100,  max_features = 'auto'))])
grid_params_rf=[{'rf__n_estimators':[100],'rf__max_features':[1,'auto'], 'rf__criterion':['gini','entropy'], 'rf__max_depth':[None, 1,2,5,10]}]
tuned_rf=sk_ms.GridSearchCV(pipeline_rf,grid_params_rf, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

ML_info['RF_pipeline'] = {'formal_name': 'Random Forest',
						  'clf': tuned_rf,
						  'calibration': 'sigmoid'}

pipeline_et = sk_pl.Pipeline([("et",sk_en.ExtraTreesClassifier())])
grid_params_et =[{'et__n_estimators':[100],'et__max_features':[1,'auto'], 'et__criterion':['gini','entropy'], 'et__max_depth':[None, 1,2,5,10]}]
tuned_et = sk_ms.GridSearchCV(pipeline_et,grid_params_et, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

ML_info['ET'] = {'formal_name': 'Extremely Randomized Trees',
				 'clf': tuned_et,
				 'calibration': 'sigmoid'}
				 
				 
pipeline_lr = sk_pl.Pipeline(steps=[('scl',sk_pp.StandardScaler()),('lr', sk_lm.LogisticRegression())])
grid_params_lr=[{'lr__penalty':['l1', 'l2'], 'lr__C':[0.1,1.,10.,100.], 'lr__solver':['saga']},
               {'lr__penalty':['elasticnet'], 'lr__l1_ratio':[0.5], 'lr__C':[0.1,1.,10.,100.], 'lr__solver':['saga']},
               {'lr__penalty':['none'], 'lr__solver':['saga']}]
tuned_lr=sk_ms.GridSearchCV(pipeline_lr,grid_params_lr, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

ML_info['LR_SCL_pipeline'] = {'formal_name': 'Logistic Regression',
					'clf': tuned_lr,
					  'calibration': 'sigmoid'}

pipeline_bt = sk_pl.Pipeline(steps=[("xgb",xgb.XGBClassifier(n_estimators=100))])
grid_params_bt=[{'xgb__n_estimators':[100], 'xgb__max_depth':[1,2,5,10], 'xgb__learning_rate':[0.05,0.1,0.25,0.5], 'xgb__gamma':[0,1,5,20]}]
tuned_bt=sk_ms.GridSearchCV(pipeline_bt,grid_params_bt, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

ML_info['BT_pipeline'] = {'formal_name': 'XGBoost',
						  'clf': tuned_bt,
						  'calibration': 'sigmoid'}
						  




