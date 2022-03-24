# In this archive we have to define the dictionary WF_info. This is a dictionary of dictionaries, that for each of our workflows
# assigns a dictionary that contains:
#
# formal_title:		Title for plots and reports
# label_name:		Variable to predict in this workflow, e.g.: 'Var17'
# feature_list:		List of features to use in the ML models, e.g.: ['Var1', 'Var2', 'Var4']
# filter_function:	Function to filter the Dataframe. If we want to use only the subject of the dataframe with Var3=1,
# 					we would write: lambda df: df.loc[df['Var3']==1].reset_index(drop=True)
# 					In case we want no filter, we have to write: lambda df: df
# group_label:		groups for cross-validation. Subjects from the same groups
#					will appear in the same folds
# validation_type: 	"kfold", "groupkfold", "stratifiedkfold", "stratifiedgroupkfold",
#					"unfilterdkfold" (for doing the kfold first and then filtering the folds)
# cv_folds:			For kfolds, the number of folds
# cv_repetitions:	For kfolds, the number of repetitions
# external_validation: 'Yes' or 'No', in case of 'Yes', you have to fill user_external_data_utils.py
#
# Example:
#
# WF_info['TallHeart'] = {'formal_title': 'Prediction of Heart Attack in tall patients',
#						  'label_name': 'Heart Attack',
#						  'feature_list': ['Age','Height','Weight','Arterial Tension'],
#						  'filter_funtion': lambda df: df.loc[df['Height']>200].reset_index(drop=True),
#						  'group_label': None,
#						  'validation_type':'stratifiedkfold',
#						  'cv_folds': 10,
#						  'cv_repetitions': 10,
#						  'external_validation': 'No'}

informativas = ['NUMERO', 'NHC','Edad en dec', 'FECHA', 'OBSERVACIÓN', 'CAUSA SUSPENSIÓN', 'ORIGEN',
				'DESTINO', 'REALIZADA']
#Podemos considerar juntar IECAS y ARA II, y también desdoblar BB y CC
variables_or = ['IMC', 'EDAD', 'Masa VI', 'TALLA', 'HASBLED', 'Creatinina',
				'TFG','DM', 'DL', 'ICC', 'EAC o CI', 'ICTUS/AIT PREVIO', 'SANGRADO PREVIO', 'NEOPLASIA', 'MOVILIDAD',
				'TEP/TVP','ANEMIA', 'RESPIRATORIO', 'DIGOXINA', 'ARM', 'SAC VAL','BB si no', 'CC s n',
				'CF>=2','CF>=3','CF==4', 'Dil AI LMS', 'Dil AI MS','Dil AI S','FEVI L-M-G','FEVI M-G', 'FEVI G', 'fumador','exfumador o fumador',
				'Genero', 'HTA', 'EAP', 'AINES', 'FA_paroxistica', 'FA_persistenteLD2',
				'Valvulopatía sign','EM', 'IM', 'Eao', 'Iao', 'IT', 'Prot mec', 'Prot bio', 'REUMATICA',
				'ACO previa si no','AAS','clopi o tica', 'Doble Antiagregantes','IECAS/ARA II',
				'HP low','HP high','Previous CVE attempt']
#DUDA: ¿está bien meter dosis bajas en precve?
variables_precve = ['sintrom','dabigatran','rivaroxaban','apixaban','edoxaban','heparina','AA pre',
					'amiodarona','flecainida','dronedarona','AVK O ACOD']
variables_cve = ['CHOQUES', 'ENERGIA MAX']
variables_poscve = ['AA post', 'amiodarona post','flecainida post','dronedarona post']
scores_cv = ['HATCH',  'CHADSVASC']
respuestas = ['CV EFECTIVA', 'RECAIDA', 'CV EXITOSA', 'CV FARMA', 'CV ESPONTÁNEA', 'CV ESP_FARM', 'ICC 2m']

eventos2m = ['ICTUS 2m', 'AIT 2m', 'ES 2m', 'IAM 2m', 'MUERTE CV 2m', 'MUERTE CC 2m', 'SANG MAYOR 2m',
		   'SANG FATAL 2m', 'SANG GI 2m', 'SANG IC 2m', 'CUALQ SANG 2m', 'ICC 2m',  'Trombo 2m',
		   'BAV/Bradi-taqui 2m', 'T. EVENTO 2 meses', 'evento2m_si_no']
eventos6m = ['ICTUS',  'AIT',  'ES',  'IAM',  'MUERTE CV',  'MUERTE CC',  'SANG MAYOR',  'SANG FATAL', 'SANG GI',
		   'SANG IC',  'CUALQ SANG',  'ICC.1',  'Trombo en algun estudio',  'BAV/Bradi-taqui', 'T. EVENTO','evento_si_no']
tiempos = ['T. CITACIÓN', 'TIEMPO ACO', 'T. RETRASO' ]

variables_desdobladas = ['CF NYHA','TABACO', 'FEVI', 'ANTIAGREGANTES', 'DAPT', 'TIPO FA',  'ANTIARRÍTMICOS PRE',
						 'ANTIARRITMICOS POST','ACO ELEGIDO', 'Diltat AI' ]
						 
variables_hatch = ['EDAD','ICC','RESPIRATORIO','ICTUS/AIT PREVIO','HTA']

variables_simples = ['EDAD','TALLA','IMC','DM', 'DL', 'ICC', 'EAC o CI', 'ICTUS/AIT PREVIO', 'MOVILIDAD','RESPIRATORIO', 'CF>=2','CF>=3','CF==4', 
					'Dil AI LMS', 'Dil AI MS','Dil AI S','FEVI L-M-G','FEVI M-G', 'FEVI G', 'fumador','exfumador o fumador','Genero', 'HTA', 'EAP', 'FA_paroxistica', 'FA_persistenteLD2','IM', 'Eao','HP low','HP high','Previous CVE attempt','AVK O ACOD']

calculadora_precve = ['FA_paroxistica','IMC', 'Dil AI LMS', 'Dil AI MS','Dil AI S','IM','FEVI L-M-G','FEVI M-G', 'FEVI G', 'CF>=2','CF>=3','CF==4', 'fumador','exfumador o fumador','Previous CVE attempt', 'ICTUS/AIT PREVIO', 'ICC','ACO previa si no', 'RESPIRATORIO', 'MOVILIDAD','BB si no', 'CC s n','IECAS/ARA II','sintrom','dabigatran','rivaroxaban','apixaban','edoxaban','heparina']



WF_info ={}



WF_info['CVSUCc'] = {'formal_title' : 'Successful CV (calculator variables)',
					'label_name' : 'CV EXITOSA',
					'feature_list' : calculadora_precve,
					'filter_function' : lambda df: df,
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'No'}
					
WF_info['CVESPc'] = {'formal_title' : 'Spontaneous Restoration of SR (calculator variables)',
					'label_name' : 'CV ESPONTÁNEA',
					'feature_list' : calculadora_precve,
					'filter_function' : lambda df: df.loc[df['AA pre']==0].reset_index(drop=True),
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'No'}

					
WF_info['CVFARc'] = {'formal_title' : 'Pharmacologic CV (calculator variables)',
					'label_name' : 'CV FARMA',
					'feature_list' : calculadora_precve,
					'filter_function' : lambda df: df.loc[df['AA pre']==1].reset_index(drop=True),
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'Yes'}
					
WF_info['AFRECc'] = {'formal_title' : 'AF Recurrence (calculator variables)',
					'label_name' : 'RECAIDA',
					'feature_list' : calculadora_precve+['EDAD']+['AA pre']+['CHOQUES','ENERGIA MAX']+['Creatinina']+['AA post', 'amiodarona post','flecainida post','dronedarona post'] +['CV ESPONTÁNEA','CV FARMA','CV EFECTIVA'],
					'filter_function' : lambda df: df.loc[df['CV EFECTIVA']==1].reset_index(drop=True),
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'Yes'}
					
WF_info['CVSUC'] = {'formal_title' : 'Successful CV',
					'label_name' : 'CV EXITOSA',
					'feature_list' : variables_or+variables_precve,
					'filter_function' : lambda df: df,
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'Yes'}


WF_info['CVEFE'] = {'formal_title' : 'Electrical CV',
					'label_name' : 'CV EFECTIVA',
					'feature_list' : variables_or+variables_precve,
					'filter_function' : lambda df: df.loc[df['CV ESP_FARM']==0].reset_index(drop=True),
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'Yes'}

					
					
WF_info['CVESP'] = {'formal_title' : 'Spontaneous Restoration of SR',
					'label_name' : 'CV ESPONTÁNEA',
					'feature_list' : variables_or+variables_precve,
					'filter_function' : lambda df: df.loc[df['AA pre']==0].reset_index(drop=True),
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'Yes'}

					
WF_info['CVFAR'] = {'formal_title' : 'Pharmacologic CV',
					'label_name' : 'CV FARMA',
					'feature_list' : variables_or+variables_precve,
					'filter_function' : lambda df: df.loc[df['AA pre']==1].reset_index(drop=True),
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'Yes'}
					
WF_info['AFREC'] = {'formal_title' : 'AF Recurrence',
					'label_name' : 'RECAIDA',
					'feature_list' : variables_or + variables_precve + ['TIEMPO ACO'] + variables_cve + variables_poscve +['CV ESPONTÁNEA']+['CV FARMA']+['CV EFECTIVA'],
					'filter_function' : lambda df: df.loc[df['CV EFECTIVA']==1].reset_index(drop=True),
					'group_label': 'NHC',
					'validation_type': 'stratifiedgroupkfold',
					'cv_folds':10,
					'cv_repetitions':10,
					'external_validation':'Yes'}
