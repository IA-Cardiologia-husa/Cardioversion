# Outcome Analysis in Elective Electrical Cardioversion of Atrial Fibrillation Patients: Development and Validation of a Machine-learning Prognostic Model

The code for the development of this Machine Learning is based in a set of luigi scripts built for this purpose. The most current version of these scripts can be found at: https://github.com/IA-Cardiologia-husa/KoopaML. An online calculator has been built with these models and can be found at https://colab.research.google.com/drive/1TbHf9waHNQYHQJhu5M9iqnpO5AESGDO5 .
To run this Python code you will need:

- luigi
- scikit learn == 0.21.3
- xgboost == 0.90
- eli5 == 0.10.1
- numpy
- pandas
- pickle
- matplotlib

The usage is:
- Start the luigi daemon: `luigid`
- Execute the luigi pipeline: `python -m luigi --module KoopaML AllTasks`, optionally with `--workers N` if you want N concurrent workers

The scripts look for a data file called *CVE-Abril2014-Diciembre2018.xlsx*, which is currently not provided. The data is processed according to the functions in *user_data_utils.py* and *user_external_data_utils.py*. The file *user_Workflow_info.py* define the different workflows, which include the ground truth label to predict, the features of the trained models and the internal validation scheme. For every workflow, the scripts will train every model defined in *user_MLmodels_info.py*, which are scikit-learn pipelines that include hyperparameter search, and feature scaling. For every workflow, the performance of risk scores, as define in *user_RiskScores_info.py* will also be calculated. Finally *user_variables_info.py* contains a dictionary of variables to do a statistical descriptive analysis.

The scripts generate:
1. A report folder, subdivided into workflows with:
    - Statistical univariant analysis of the variables
    - ROC curve of the trained models (internal and external validation)
    - PR curve of the trained models (internal and external validation)
    - A BestML_model_report with numerical values for the area under the ROC/PR curves and its confidence interval, the value of operational thresholds automatically selected with different criteria and the respective values of sensitivity and specificity, and the feature importances as measured by the MDA algorithm using the eli5 libray
2. A model folder, with all the models trained (in pickle format) and an excel file with the hyperparameter selection results.
3. An intermediate folder with all the intermediate files generated
4. A log folder with the log of the luigi execution

---
Currently there are 10 workflows defined. The 5 original workflows as shown in the article, and 5 additional workflows with a restricted set of features (selected from the most important ones as measured by the MDA algorithm) to use in the online calculator.

A roc_data folder contains the intermediate results needed to plot ROC curves in the online calculator.


