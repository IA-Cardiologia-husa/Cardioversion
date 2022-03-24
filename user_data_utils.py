#Importación de librerías
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def load_database():
	df_excel = pd.read_excel("CVE-Abril2014-Diciembre2018.xlsx", sheet_name=1)
	df_excel = df_excel.drop(df_excel.shape[0]-1) # Borrar última fila con descripciones

	return df_excel

def clean_database(df_excel):
	return df_excel

def process_database(df_excel):

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
	
	df_excel=df_excel.loc[df_excel['FECHA'].dt.year < 2018,:]

	return df_excel

def fillna_database(df_excel):

	#Rellenamos CV en casos que no aplican

	df_excel.loc[df_excel['CV EFECTIVA']==0,'RECAIDA']=0
	df_excel.loc[df_excel['AA pre']==0, 'CV FARMA']=0
	df_excel.loc[df_excel['AA pre']==1, 'CV ESPONTÁNEA']=0

	#Los pacientes con Insuficiencia tricuspídea y sin valor de hipertensión pulmonar, tomamos la media
	#de la hipertensión pulmonar con insuficiencia tricuspidea. Igual con los que no lo tienen
	df_excel.loc[(df_excel['IT']==1)&(df_excel['HP'].isnull()),'HP'] = df_excel.loc[(df_excel['IT']==1)&(~df_excel['HP'].isnull()),'HP'].mean()
	df_excel.loc[(df_excel['IT']==0)&(df_excel['HP'].isnull()),'HP'] = df_excel.loc[(df_excel['IT']==0)&(~df_excel['HP'].isnull()),'HP'].mean()

	#Los que falta peso y talla, rellenamos con la media del peso y talla de las personas del mismo genero
	df_excel.loc[df_excel['PESO'].isnull()&df_excel['TALLA'].isnull(),['NUMERO','PESO','TALLA','Genero']]

	falta_peso_talla_m = df_excel['PESO'].isnull()&df_excel['TALLA'].isnull()&(df_excel['Genero']==1)
	falta_peso_talla_f = df_excel['PESO'].isnull()&df_excel['TALLA'].isnull()&(df_excel['Genero']==0)

	media_peso_m = df_excel.loc[(df_excel['Genero']==1)&(~df_excel['PESO'].isnull()),'PESO'].mean()
	media_peso_f = df_excel.loc[(df_excel['Genero']==0)&(~df_excel['PESO'].isnull()),'PESO'].mean()
	media_talla_m = df_excel.loc[(df_excel['Genero']==1)&(~df_excel['TALLA'].isnull()),'TALLA'].mean()
	media_talla_f = df_excel.loc[(df_excel['Genero']==0)&(~df_excel['TALLA'].isnull()),'TALLA'].mean()
	media_imc_m = 10000*media_peso_m/media_talla_m**2
	media_imc_f = 10000*media_peso_f/media_talla_f**2

	df_excel.loc[falta_peso_talla_m,'PESO'] = media_peso_m
	df_excel.loc[falta_peso_talla_f,'PESO'] = media_peso_f
	df_excel.loc[falta_peso_talla_m,'TALLA'] = media_talla_m
	df_excel.loc[falta_peso_talla_f,'TALLA'] = media_talla_f

	#Las personas que solo le falta el peso, o la talla, rellenamos también con la media
	#Pensé en rellenar la talla de modo que quede el IMC igual que la media de su genero, pero eso puede dar
	#lugar a outliers (150 kilos de peso sin talla -> 226cm de altura)

	falta_peso_m = df_excel['PESO'].isnull()&~(df_excel['TALLA'].isnull())&(df_excel['Genero']==1)
	falta_peso_f = df_excel['PESO'].isnull()&~(df_excel['TALLA'].isnull())&(df_excel['Genero']==0)
	falta_talla_m = df_excel['TALLA'].isnull()&~(df_excel['PESO'].isnull())&(df_excel['Genero']==1)
	falta_talla_f = df_excel['TALLA'].isnull()&~(df_excel['PESO'].isnull())&(df_excel['Genero']==0)

	df_excel.loc[falta_peso_m,'PESO'] = media_imc_m*df_excel.loc[falta_peso_m,'TALLA']**2/10000
	df_excel.loc[falta_peso_f,'PESO'] = media_imc_f*df_excel.loc[falta_peso_f,'TALLA']**2/10000

	df_excel.loc[falta_talla_m,'TALLA'] = media_talla_m
	df_excel.loc[falta_talla_f,'TALLA'] = media_talla_f

	#Rellenamos el IMC con las medias por cada género

	df_excel.loc[df_excel['IMC'].isnull()&(df_excel['Genero']==1),'IMC'] = media_imc_m
	df_excel.loc[df_excel['IMC'].isnull()&(df_excel['Genero']==0),'IMC'] = media_imc_f

	#Rellenamos los choques y la energia con la mediana y la media
	df_excel.loc[(df_excel['CV ESP_FARM']==0)&~(df_excel['CHOQUES']>0),'CHOQUES']=\
		df_excel.loc[(df_excel['CV ESP_FARM']==0)&(df_excel['CHOQUES']>0),'CHOQUES'].median()
	df_excel.loc[(df_excel['CV ESP_FARM']==0)&~(df_excel['ENERGIA MAX']>0),'ENERGIA MAX'] =\
		df_excel.loc[(df_excel['CV ESP_FARM']==0)&(df_excel['ENERGIA MAX']>0),'ENERGIA MAX'].mean()

	#Los que le falta la masa indexada pero tenemos la masa, la calculamos
	df_excel['BSA'] = np.sqrt(df_excel['TALLA']*df_excel['PESO']/3600)
	df_excel.loc[df_excel['Masa'].notnull()&df_excel['Masa VI'].isnull(),'Masa VI'] = df_excel.loc[df_excel['Masa'].notnull()&df_excel['Masa VI'].isnull(),'Masa'] / df_excel.loc[df_excel['Masa'].notnull()&df_excel['Masa VI'].isnull(),'BSA']
	
	
	for i in ['EDAD','Vol AI', 'Masa', 'Masa VI', 'Creatinina', 'TFG', 'T. CITACIÓN', 'T. RETRASO']:
		df_excel.loc[df_excel[i].isnull(),i]=df_excel[i].mean()

	for i in ['ARM', 'AVK O ACOD']:
		df_excel.loc[df_excel[i].isnull(),i]=df_excel[i].median()


	#Las columnas con valores NaN que quedan, se pueden rellenar con un 0
	for i in ['INTENTOS DE CVE', 'CHOQUES', 'ENERGIA MAX']:
		if(df_excel[i].isnull().any()):
			df_excel[i]=df_excel[i].fillna(0)

	#Las columnas con enteros que tuvieran nans eran de tipo object, pasamos todo a int64

	for i in df_excel.columns:
		if(df_excel[i].dtype== np.dtype('object')):
			if(~df_excel[i].isnull().any()):
				df_excel[i]=df_excel[i].astype('int64')

	return df_excel

def preprocess_filtered_database(df, wf_name):
	return df
