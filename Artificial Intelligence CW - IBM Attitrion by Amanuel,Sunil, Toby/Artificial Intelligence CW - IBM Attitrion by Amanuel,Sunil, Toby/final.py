"""
How to Run:
    1. Install package: pip install -U imbalanced-learn 
    2. Restart working environment (Close Spyder and run again) 
    3. Repeat the process to ensure packages were installed correctly and Spyder restarted correctly.
    
Run this on Spyder console or a Python Console. 

This was tested several times accross different computes to ensure that, it is in fact executable.

Lines 273,274,276 are commented, so that the search isn't run.  
Performing a search for the best parameters takes roughly more than 30mins. 

This program operates with the best parameters that was provided by the search. 

References are provided at the bottom. 

IBM Attitrion by Amanuel, Sunil and Toby
"""
import pandas as pd
import os
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import matplotlib.pyplot as plt

from imblearn.over_sampling import (SMOTE, 
                                    ADASYN)
from imblearn.under_sampling import NearMiss

path = "."  #absolute or relative path to the folder containing the file. 
            #"." for current folder
filename_read = os.path.join(path, "IBM-HR-Employee-Attrition.csv")

#   Making a list of missing value types
missing_values = ["n/a", "na", "--", '?', 'NA']


df = pd.read_csv(filename_read, na_values= missing_values)
print("Missing values have been handled")

#   Check if is there any null values within our dataset
naValues = df.isna().sum().sum()
print("Check for NA values in dataset:"+ str(df.isna().sum().sum()))

#   check if there is any duplication
print("Check for duplications in dataset: "+ str(sum(df.duplicated())))

# correlation matrix of our dataset
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

#   Removing unnecessary columns that don't affect our analysis and some have default values. 
#   e.g. All employees have the same standardhours and are over 18+ years old
df = df.drop(columns=['StandardHours', 
                          'EmployeeCount', 
                          'EmployeeNumber',
                          'Over18',
                        ])
#Dropped unnecessary columns

print()
print("Removing highly correlated columns with thresold set to 70%")

#   Removing highly correlated feature, the threshold for deciding is set to 70%. 
#   Threshold of 0.7% was decided as anymore above from 75% only removed 1 columns. Anything under 65% removed 7 columns and too low to draw to suggest high correlations. 
#   Therefore 0.7% is the optimal value therefore to draw conclusion.   
threshold = 0.7
# Get the correlation values from correlation matrix
corr_matrix = df.corr().abs()
corr_matrix.head()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove :' % (len(to_drop)))
df = df.drop(columns = to_drop)
print("Dropped highly correlated columns")
print(to_drop)


# Pie chart to represent skew in target variable "Attrition"
plt.figure(figsize=[4,4])
src=df.Attrition.value_counts()
plt.pie(src,labels=src.index,startangle=90,counterclock=False,colors=['#8db3ef','#f9e5f9'],
        wedgeprops={'width':0.6},autopct='%1.1f%%', pctdistance=0.75);

plt.show()  

# Encoding
# Data is split into categorical and numerical. 
# Categorical data is encoded into numerical values and then concatenated with the numerical dataset

# Empty list to store columns with categorical data
categorical = []    
for col, value in df.iteritems():
    if value.dtype == 'object':
        categorical.append(col)
        
# Store the numerical columns in a separate list
numerical = df.columns.difference(categorical)

# Store the categorical data in a separate dataframe df_cat
df_cat = df[categorical]

#   Dropping Attrition (target variable) column as it will be encoded differently, using a dictionary later down the line
df_cat = df_cat.drop(['Attrition'], axis=1)

#   Categorical data is encoded using panda.get dummies
df_cat = pd.get_dummies(df_cat)

# Store the numerical features to a separate dataframe df_num
df_num = df[numerical]

# Concat the two dataframes together columnwise to create a complete dataseet
df_final  = pd.concat([df_num, df_cat], axis = 1)

# Define a dictionary for the target mapping, this is used instead of using pandas encoding our target variable
# variable into two separate column. It swaps Yes with 1s and No with 0s.
target_map = {'Yes':1, 'No':0}

# Use the pandas apply method to numerically encode our attrition target variable
target = df["Attrition"].apply(lambda x: target_map[x])


# Split data into train and test sets as well as for testing
X_train, X_test, y_train, y_test = train_test_split(df_final, 
                                                         target, 
                                                         train_size= 0.80,
                                                         random_state=0);
# oversampling should be done after dividing the data to prevent information leaking


                                                    
decision_classifer = DecisionTreeClassifier

# After splitting, pipeline are used to handle several task in a single line 
# Decision classior alongisde various oversample/undersample techniques, to gain peformance on
# the vaarious sampling techniques ,decision tree classifer is used as it is the easiest to implement     
                                               
print()
#building a decision tree model with a original unbalanced distribution
decisiontreepipeline = make_pipeline_imb(decision_classifer(random_state=0))
decisiontreemodel = decisiontreepipeline.fit(X_train, y_train)
decisiontreeprediction = decisiontreemodel.predict(X_test)
print("Performance of decision tree with the original unbalanced distribution: "+ format(decisiontreepipeline.score(X_test, y_test)))

# decision tree + SMOTE oversampling technique on our dataset
smote_pipeline = make_pipeline_imb(SMOTE(),decision_classifer(random_state=0))
smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)
print("Performance of decision tree with the oversampling technique SMOTE: "+ format(smote_pipeline.score(X_test, y_test)))             

# decision tree + ADASYN oversampling technique on our dataset
adasyn_pipeline = make_pipeline_imb(ADASYN(), decision_classifer(random_state=0))
adasyn_model = adasyn_pipeline.fit(X_train, y_train)
adasyn_prediction = adasyn_model.predict(X_test)
print("Performance of decision tree with the oversampling technique ADASYN: "+ format(adasyn_pipeline.score(X_test, y_test)))

# decision tree + NearMiss undersampling technique on our dataset
nearmiss_pipeline = make_pipeline_imb(NearMiss(), decision_classifer(random_state=0))
nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
nearmiss_prediction = nearmiss_model.predict(X_test)
print("Performance of decision tree with the undersampling technique NearMiss: "+ format(nearmiss_pipeline.score(X_test, y_test)))


print()
print("Chart to show the performance metrics of these sampling techniqes")
groups = [[accuracy_score(y_test, smote_prediction),precision_score(y_test, smote_prediction),recall_score(y_test, smote_prediction)], 
          [accuracy_score(y_test, adasyn_prediction),precision_score(y_test, adasyn_prediction),recall_score(y_test, adasyn_prediction)], 
          [accuracy_score(y_test, nearmiss_prediction),precision_score(y_test, nearmiss_prediction),recall_score(y_test, nearmiss_prediction)]]
group_labels = ['SMOTE', 'ADASYN', 'NearMiss']

# Convert data to pandas DataFrame.
df = pd.DataFrame(groups, index=group_labels).T

# Plot.
pd.concat(
    [df.mean().rename('precision'), df.min().rename('recall'), 
     df.max().rename('accuracy')],
    axis=1).plot.barh()
plt.ylabel('Oversampling & Undersampling')
plt.xlabel('Score')
plt.show()


print()
print("SMOTE had been chosen to tackle the imbalance in the dataset. It provided better recall, precision and accuracy than the alternatives.")    

# Oversampling our training set - SMOTE
smote_oversampler=SMOTE(random_state=0)
smote_train, smote_target = smote_oversampler.fit_sample(X_train,y_train)

                                         
print()
print("Training different models with 10 being the base estimator value, while using the SMOTE sampling technique")
print()

randomforest_classifier = RandomForestClassifier
ada_classifier = AdaBoostClassifier
gradientBoost_classifier = GradientBoostingClassifier

#Default value of estimators 10 is decided. Different version of spyder have different default values.
# To ensure, that our results aren't different across different versions of spyder. Random Rate and Estimators have been set to predetermined values. 

print("Decision Tree results: "+ format(smote_pipeline.score(X_test, y_test))) #Previous code of SMOTE + Decision Tree

# build model with SMOTE  + RandomForest
smote_pipelineRF = make_pipeline_imb(SMOTE(random_state=0), randomforest_classifier(n_estimators = 10, random_state=0))
smote_modelRF = smote_pipelineRF.fit(X_train, y_train)
smote_predictionRF = smote_modelRF.predict(X_test)
print("Random Forest: "+ format(smote_pipelineRF.score(X_test, y_test)))

# build model with SMOTE  + AdaBoost
smote_pipelineADA = make_pipeline_imb(SMOTE(random_state=0), ada_classifier(n_estimators = 10,random_state=0))
smote_modelADA = smote_pipelineADA.fit(X_train, y_train)
smote_predictionADA = smote_modelADA.predict(X_test)
print("AdaBoost: "+ format(smote_pipelineADA.score(X_test, y_test)))

# build model with SMOTE  + GradientBoost
smote_pipelineGB = make_pipeline_imb(SMOTE(random_state=0), gradientBoost_classifier(n_estimators = 10,random_state=0))
smote_modelGB = smote_pipelineGB.fit(X_train, y_train)
smote_predictionGB = smote_modelGB.predict(X_test)
print("GradientBoost: "+ format(smote_pipelineGB.score(X_test, y_test)))

# Plot to show a comparision of different models alongside their performance metrics
plt.show()
groups = [[accuracy_score(y_test, smote_predictionADA),precision_score(y_test, smote_predictionADA),recall_score(y_test, smote_predictionADA)], 
          [accuracy_score(y_test, smote_predictionGB),precision_score(y_test, smote_predictionGB),recall_score(y_test, smote_predictionGB)], 
          [accuracy_score(y_test, smote_predictionRF),precision_score(y_test, smote_predictionRF),recall_score(y_test, smote_predictionRF)],
          [accuracy_score(y_test, smote_prediction),precision_score(y_test, smote_prediction),recall_score(y_test, smote_prediction)]]
group_labels = ['AdaBoost', 'GradientBoost', 'RandomForest', 'DecisionTree']


# Convert data to pandas DataFrame.
df = pd.DataFrame(groups, index=group_labels).T
# Plot.
pd.concat(
    [df.mean().rename('recall'), df.min().rename('precision'), 
     df.max().rename('accuracy')],
    axis=1).plot.barh()
plt.ylabel('Models')
plt.xlabel('Score')
plt.show()


print()
print("ADABoost is the best model to use as results show that it has a better precision (0.44) and recall (0.59) compared to other models.")


# Fine tuning hyperparameters using GridSearchCV
# Built our classifier 
best_classifier = AdaBoostClassifier(random_state=0)

# Hyperparameters that can be fine tuned
# AdaBoost has a default of n_estimators of 50, with learning_rate of 1.  
# GridSearchCV starts with an estimator of 50 to 300, incrementing by 10
# Learning_rate has a default of 1. THe GridSearchCV starts from 1 to 2, incrementing by 0.1

param_grid = { 
    'n_estimators': np.arange(40,310,10),
    'learning_rate': np.arange(0.9,2.1, 0.1)  
}

print('Running a GridSearch for the best fine tuning')

#CV_rfc = GridSearchCV(best_classifier, param_grid=param_grid, cv= 5)
#CV_rfc.fit(smote_train, smote_target)
#Outputting GridSearchCV best parameters
#print(CV_rfc.best_params_)  

# Building a model using the best parameters
clf = AdaBoostClassifier(n_estimators=260, learning_rate=0.9, random_state=0)
clf = clf.fit(smote_train,smote_target)
y_pred = clf.predict(X_test)


print()
print("Comparision of default Adaboost vs optimised AdaBoost")

# Method to print performance metrics of a model
def print_results(headline, true_value, pred):
   print(headline)
   print("accuracy: {}".format(accuracy_score(true_value, pred)))
   print("precision: {}".format(precision_score(true_value, pred)))
   print("recall: {}".format(recall_score(true_value, pred)))
   print("f1: {}".format(f1_score(true_value, pred)))
print()

#Outputting the performance metrics of Adaboost and Best paramaters Adaboost
print_results("Default AdaBoost results:", y_test,smote_predictionADA)
print()

print_results("Best parameters AdaBoost results:", y_test, y_pred)

#Chart to present the top 10 features to attritrion 
pd.Series(clf.feature_importances_, 
         index=df_final.columns).sort_values(ascending=False).nlargest(10).plot(kind='bar', figsize=(18,6));
plt.title('Top 10 Important Features to Attrition')
plt.ylabel('Scores')
plt.xlabel('Features')
plt.show()


#Reference 
#Dataset : https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

#Encoding & building models: https://www.kaggle.com/arthurtok/employee-attrition-via-ensemble-tree-based-methods 

#For GridSearchCV and Features Importance
#https://datascience.stackexchange.com/questions/13754/feature-importance-with-scikit-learn-random-forest-shows-very-high-standard-devi

#Oversampling & Undersampling & Using Pipeline for running several tasks
# https://towardsdatascience.com/yet-another-twitter-sentiment-analysis-part-1-tackling-class-imbalance-4d7a7f717d44
# https://github.com/coding-maniacs/over-under-sampling/blob/master/src/main.py 