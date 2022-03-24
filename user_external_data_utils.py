#Import libraries
import pandas as pd
import numpy as np
from user_data_utils import *

def load_external_database():
	# df = pd.read_excel("External Database.xlsx")
	df_excel = pd.read_excel("CVE-Abril2014-Diciembre2018.xlsx", sheet_name=1)
	df_excel = df_excel.drop(df_excel.shape[0]-1) # Borrar última fila con descripciones
	
	return df_excel

def clean_external_database(df_excel):
	df_excel = clean_database(df_excel)
	return df_excel

def process_external_database(df_excel):
	#Desdoblamos variables y las agrupamos en función del momento

	df_excel['evento2m_si_no'] = (df_excel['T. EVENTO 2 meses'] != 0)*1
	df_excel['evento_si_no'] = (df_excel['T. EVENTO'] != 0)*1

	#NOTAS:
	#TABACO en 2 variables dummies
	df_excel['fumador']=0
	# df_excel['exfumador']=0
	df_excel['exfumador o fumador']=0
	df_excel.loc[(df_excel['TABACO']==1),'fumador']=1
	df_excel.loc[(df_excel['TABACO']==1),'exfumador o fumador']=1
	df_excel.loc[(df_excel['TABACO']==2),'exfumador o fumador']=1

	#'AVK o ACOD' lo pasamos a 0 y 1 en vez de 1 y 2
	df_excel.loc[(df_excel['AVK O ACOD'] == 2), 'AVK O ACOD'] = 0
	
	# Creamos el opuesto para el descriptivo
	df_excel['ACOD'] = 1-df_excel['AVK O ACOD']

	#Genero lo pasamos a 0 y 1 en vez de 1 y 2
	df_excel.loc[(df_excel['Genero'] == 2), 'Genero'] = 0

	#Vamos a llevar hiperstensión pulmonar a variables categóricas (https://dx.doi.org/10.1016%2Fj.ijcha.2016.05.011)

	df_excel['HP low'] = 0
	df_excel['HP high'] = 0
	df_excel.loc[df_excel['HP']>280, 'HP low'] = 1
	df_excel.loc[df_excel['HP']>340, 'HP high'] = 1


	# Antiagregantes y DAPT los transformo en 4 variables dummies. (no hay prasugrel en la base de datos)
	df_excel['AAS']=0
	df_excel['clopi o tica']=0
	df_excel['Doble Antiagregantes']=0
	df_excel.loc[(df_excel['ANTIAGREGANTES']==1), 'AAS'] = 1
	df_excel.loc[(df_excel['ANTIAGREGANTES']>1), 'clopi o tica'] = 1
	df_excel.loc[(df_excel['DAPT']>0), 'clopi o tica'] = 1
	df_excel.loc[(df_excel['DAPT']>0), 'Doble Antiagregantes'] = 1


	# 'Tipo FA' la convertimos en 3 variables dummies, usamos FA_paroxistica y persistenteLD2 en las variables
	df_excel['FA_paroxistica'] = 0
	df_excel['FA_persistente1'] = 0
	df_excel['FA_persistenteLD2'] = 0
	df_excel.loc[(df_excel['TIPO FA']==0),'FA_paroxistica'] = 1
	df_excel.loc[(df_excel['TIPO FA']==1),'FA_persistente1'] = 1
	df_excel.loc[(df_excel['TIPO FA']==2),'FA_persistenteLD2'] = 1

	#Indice de masa corporal
	df_excel['IMC']=10000*df_excel['PESO']/df_excel['TALLA']**2

	#Vamos a dividir FEVI en 3 variables (leve, moderada, grave) FEVI L-M-G, FEVI M-G (que ya existe), FEVI G
	df_excel['FEVI L-M-G']=0
	df_excel['FEVI G']=0
	df_excel.loc[(df_excel['FEVI']>0),'FEVI L-M-G']=1
	df_excel.loc[(df_excel['FEVI']==3),'FEVI G']=1


	#Vamos a dividir CF NYHA en 3 variables
	df_excel['CF>=2']=0
	df_excel['CF>=3']=0
	df_excel['CF==4']=0
	df_excel.loc[(df_excel['CF NYHA']>1),'CF>=2']=1
	df_excel.loc[(df_excel['CF NYHA']>2),'CF>=3']=1
	df_excel.loc[(df_excel['CF NYHA']>3),'CF==4']=1

	#Vamos a dividir los diversos anticoagulantes
	#Cuando elegimos un anticoagulantes ¿quitamos el que estaba tomando antes?
	df_excel['sintrom']=0
	df_excel['dabigatran']=0
	df_excel['rivaroxaban']=0
	df_excel['apixaban']=0
	df_excel['edoxaban']=0
	df_excel['heparina']=0
	df_excel.loc[(df_excel['ACO ELEGIDO']==1),'sintrom']=1
	df_excel.loc[(df_excel['ACO ELEGIDO']==2),'dabigatran']=1
	df_excel.loc[(df_excel['ACO ELEGIDO']==3),'rivaroxaban']=1
	df_excel.loc[(df_excel['ACO ELEGIDO']==4),'apixaban']=1
	df_excel.loc[(df_excel['ACO ELEGIDO']==5),'edoxaban']=1
	df_excel.loc[(df_excel['ACO ELEGIDO']==6),'heparina']=1

	#Vamos a dividir los antiarrítmicos
	df_excel['amiodarona']=0
	df_excel['flecainida']=0
	df_excel['dronedarona']=0
	#No hace falta "otros" porque no pueden tener dos antiarrítimicos a la vez y ya tenemos una variables dicotómica 'AA pre' de si
	#toma antiarrítmicos o no
	df_excel.loc[(df_excel['ANTIARRÍTMICOS PRE']==1),'amiodarona']=1
	df_excel.loc[(df_excel['ANTIARRÍTMICOS PRE']==2),'flecainida']=1
	df_excel.loc[(df_excel['ANTIARRÍTMICOS PRE']==3),'dronedarona']=1

	#Igual con los antiarrítmicos post
	df_excel['amiodarona post']=0
	df_excel['flecainida post']=0
	df_excel['dronedarona post']=0
	#No hace falta "otros" porque no pueden tener dos antiarrítimicos a la vez y ya tenemos una variables dicotómica 'AA post' de si
	#toma antiarrítmicos o no
	df_excel.loc[(df_excel['ANTIARRITMICOS POST']==1),'amiodarona post']=1
	df_excel.loc[(df_excel['ANTIARRITMICOS POST']==2),'flecainida post']=1
	df_excel.loc[(df_excel['ANTIARRITMICOS POST']==3),'dronedarona post']=1

	#Escribimos el score HATCH
	df_excel['HATCH']=0
	df_excel.loc[(df_excel['EDAD']>75),'HATCH']+=1
	df_excel.loc[(df_excel['ICC']==1),'HATCH']+=2
	df_excel.loc[(df_excel['RESPIRATORIO']==1),'HATCH']+=1
	df_excel.loc[(df_excel['ICTUS/AIT PREVIO']==1),'HATCH']+=2
	df_excel.loc[(df_excel['HTA']==1),'HATCH']+=1
    
	df_excel['HATCH_OR']=0
	df_excel.loc[(df_excel['EDAD']>75),'HATCH_OR']+=0.45
	df_excel.loc[(df_excel['ICC']==1),'HATCH_OR']+=0.80
	df_excel.loc[(df_excel['RESPIRATORIO']==1),'HATCH_OR']+=0.41
	df_excel.loc[(df_excel['ICTUS/AIT PREVIO']==1),'HATCH_OR']+=0.71
	df_excel.loc[(df_excel['HTA']==1),'HATCH_OR']+=0.42


	#Vamos a llevar hiperstensión pulmonar a variables categóricas (https://dx.doi.org/10.1016%2Fj.ijcha.2016.05.011)

	df_excel['HP low'] = 0
	df_excel['HP high'] = 0
	df_excel.loc[df_excel['HP']>280, 'HP low'] = 1
	df_excel.loc[df_excel['HP']>340, 'HP high'] = 1

	#Vamos a añadir dilatación severa de la aurícula izquierda
	df_excel['Dil AI LMS']=0
	df_excel['Dil AI LMS'] = df_excel['Vol AI']>=35
	df_excel['Dil AI MS'] = df_excel['Vol AI']>=42
	df_excel['Dil AI S'] = df_excel['Vol AI']>=48

	df_excel.loc[df_excel['CV EFECTIVA']==0,'RECAIDA']=np.nan
	df_excel.loc[df_excel['RECAIDA'].isnull()&(df_excel['CV EFECTIVA']==1),'RECAIDA']=1-df_excel.loc[df_excel['RECAIDA'].isnull()&(df_excel['CV EFECTIVA']==1),'CV EXITOSA']

	df_excel.loc[df_excel['CV EFECTIVA']==0,'RECAIDA']=np.nan
	df_excel.loc[df_excel['AA pre']==0, 'CV FARMA']=np.nan
	df_excel.loc[df_excel['AA pre']==1, 'CV ESPONTÁNEA']=np.nan

	#Unimos IECAs y ARAII
	df_excel['IECAS/ARA II'] = df_excel['IECAS'] | df_excel['ARA II']
	df_excel['age75'] = (df_excel['EDAD']>75)
	df_excel['age65'] = (df_excel['EDAD']>65)
	
	df_excel=df_excel.loc[df_excel['FECHA'].dt.year >= 2018,:]
	return df_excel

def fillna_external_database(df_excel):
	df_excel = fillna_databse(df_excel)
	return df_excel

def preprocess_filtered_external_database(df, wf_name):
	return df