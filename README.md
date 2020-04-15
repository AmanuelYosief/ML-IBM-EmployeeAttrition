# Machine Learning Python
 University coursework. Using machine learning to predict IBM attrition and explore features that could lead to increased Attitrion Rates

Spyder(Python) outputs:

Missing values have been handled
Check for NA values in dataset:0
Check for duplications in dataset: 0

![GitHub Logo](/images/confusionmatrix.png)

Removing highly correlated columns with thresold set to 70%
There are 5 columns to remove :
Dropped highly correlated columns
['MonthlyIncome', 'PerformanceRating', 'TotalWorkingYears', 'YearsInCurrentRole', 'YearsWithCurrManager']


![GitHub Logo](/images/piechart.png)

Performance of decision tree with the original unbalanced distribution: 0.8027210884353742
Performance of decision tree with the oversampling technique SMOTE: 0.7517006802721088
Performance of decision tree with the oversampling technique ADASYN: 0.7585034013605442
Performance of decision tree with the undersampling technique NearMiss: 0.6190476190476191

Chart to show the performance metrics of these sampling techniqes


![GitHub Logo](/images/samplingmethods.png)

SMOTE had been chosen to tackle the imbalance in the dataset. It provided better recall, precision and accuracy than the alternatives.

Training different models with 10 being the base estimator value, while using the SMOTE sampling technique

Decision Tree results: 0.7517006802721088
Random Forest: 0.8435374149659864
AdaBoost: 0.8095238095238095
GradientBoost: 0.8095238095238095


![GitHub Logo](/images/models.png)

ADABoost is the best model to use as results show that it has a better precision (0.44) and recall (0.59) compared to other models.
Running a GridSearch for the best fine tuning

Comparision of default Adaboost vs optimised AdaBoost

Default AdaBoost results:
accuracy: 0.8095238095238095
precision: 0.4461538461538462
recall: 0.5918367346938775
f1: 0.5087719298245614

Best parameters AdaBoost results:
accuracy: 0.8401360544217688
precision: 0.525
recall: 0.42857142857142855
f1: 0.4719101123595506


![GitHub Logo](/images/topfeatures.png)



