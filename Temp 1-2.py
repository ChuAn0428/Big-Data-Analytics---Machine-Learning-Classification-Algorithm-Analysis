# -*- coding: utf-8 -*-
'''
Author: Chu-An Tsai
CS 59000 Big Data Analytics
Final project

Temp 1-2
'''
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas_profiling  as pp
from sklearn import tree
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn import linear_model
from sklearn.svm import LinearSVC
import xgboost as xgb
from xgboost import plot_importance
import math
from sklearn.metrics import plot_roc_curve
from time import process_time


df = pd.read_csv("dataset/cardio.csv", delimiter=";")
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 2)
print('\n1. Dataset description (Original):')
print(df.describe())

df['age'] = round(df['age']/365).apply(lambda x: int(x))
df['ap_hi'] = abs(df['ap_hi']).apply(lambda x: int(x))
df['ap_lo'] = abs(df['ap_lo']).apply(lambda x: int(x))

df.drop(df[(df['height'] > df['height'].quantile(0.995)) | (df['height'] < df['height'].quantile(0.005))].index,inplace=True)
df.drop(df[(df['weight'] > df['weight'].quantile(0.995)) | (df['weight'] < df['weight'].quantile(0.005))].index,inplace=True)

df = df[df['ap_hi'] > df['ap_lo']].reset_index(drop=True)

df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.995)) | (df['ap_hi'] < df['ap_hi'].quantile(0.005))].index,inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.995)) | (df['ap_lo'] < df['ap_lo'].quantile(0.005))].index,inplace=True)


print('\n2. Dataset description (After clearning):')
print(df.describe())
#print('\n3. Full report of the dataset (After adjustment):')
#print(pp.ProfileReport(df))

"""
X=df.drop(['cardio','id'], axis=1)
Y=df['cardio']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
features_list = X.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(5,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()
"""

data = df.values.copy()
#x = data[:,1:-1].copy()

pca = PCA(n_components=4)
x = pca.fit_transform(data[:,1:-1].copy())  

print('\nPCA ratio(used):',pca.explained_variance_ratio_)

y = data[:,-1].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

def conf_cal(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
           TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
           FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
           TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
           FN += 1

    return TP, FP, TN, FN

target_names = ['Negative (-)', 'Positive (+)']


##### Naive Bayes ######
time_start = process_time()    

nb_gnb = GaussianNB().fit(x_train, y_train)
nb_pred = nb_gnb.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, nb_pred)
rep_nb = classification_report(y_test, nb_pred, target_names=target_names)
acc_nb = accuracy_score(y_test, nb_pred)

print("\n1. Naive Bayes:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_nb)
acc_nb2 = cross_val_score(nb_gnb, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_nb2,4))

nb_disp = plot_roc_curve(nb_gnb, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
nb_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')

##### Decision Tree #####
time_start = process_time()   

dtree = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=4).fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, dtree_pred)
rep_dtree = classification_report(y_test, dtree_pred, target_names=target_names)
acc_dtree = accuracy_score(y_test, dtree_pred)

print("\n2. Decision Trees:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_dtree)
acc_dtree2 = cross_val_score(dtree, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_dtree2,4))

dtree_disp = plot_roc_curve(dtree, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
dtree_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')

##### K-Nearest Neighbors #####
time_start = process_time()   
knn = KNeighborsClassifier(n_neighbors=18).fit(x_train, y_train)
knn_pred = knn.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, knn_pred)
rep_knn = classification_report(y_test, knn_pred, target_names=target_names)
acc_knn = accuracy_score(y_test, knn_pred)

print("\n3. K-Nearest Neighbors:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_knn)
acc_knn2 = cross_val_score(knn, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_knn2,4))

knn_disp = plot_roc_curve(knn, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
knn_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')


##### K-Nearest Neighbors (Bagging) #####
time_start = process_time()   

knn_bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=18),max_samples=0.5, max_features=0.5).fit(x_train, y_train)
knn_bagging_pred = knn_bagging.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, knn_bagging_pred)
rep_knn_bagging = classification_report(y_test, knn_bagging_pred, target_names=target_names)
acc_knn_bagging = accuracy_score(y_test, knn_bagging_pred)

print("\n4. K-Nearest Neighbors (Bagging):")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_knn_bagging)

acc_knn_bagging2 = cross_val_score(knn_bagging, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_knn_bagging2,4))

knn_bagging_disp = plot_roc_curve(knn_bagging, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
knn_bagging_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')


##### Logistic Regression #####
time_start = process_time()   

lr = LogisticRegression(random_state=0, max_iter=10000).fit(x_train, y_train)
lr_pred = lr.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, lr_pred)
rep_lr = classification_report(y_test, lr_pred, target_names=target_names)
acc_lr = accuracy_score(y_test, lr_pred)

print("\n5. Logistic Regression:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_lr)

acc_lr2 = cross_val_score(lr, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_lr2,4))

lr_disp = plot_roc_curve(lr, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
lr_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')


##### Random Forest #####
time_start = process_time()   

rfc = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=2, random_state=0).fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, rfc_pred)
rep_rfc = classification_report(y_test, rfc_pred, target_names=target_names)
acc_rfc = accuracy_score(y_test, rfc_pred)

print("\n6. Random Forest:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_rfc)

acc_rfc2 = cross_val_score(rfc, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_rfc2,4))

rfc_disp = plot_roc_curve(rfc, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
rfc_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')

##### Adaboost #####
time_start = process_time()   

adb = AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)
adb_pred = adb.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, adb_pred)
rep_adb = classification_report(y_test, adb_pred, target_names=target_names)
acc_adb = accuracy_score(y_test, adb_pred)

print("\n7. Adaboost:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_adb)

acc_adb2 = cross_val_score(adb, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_adb2,4))

adb_disp = plot_roc_curve(adb, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
adb_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')

##### Gradient Boosting Decision Tree #####
time_start = process_time()   

gbdt = GradientBoostingClassifier(n_estimators=100).fit(x_train, y_train)
gbdt_pred = gbdt.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, gbdt_pred)
rep_gbdt = classification_report(y_test, gbdt_pred, target_names=target_names)
acc_gbdt = accuracy_score(y_test, gbdt_pred)

print("\n8. Gradient Boosting Decision Tree:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_gbdt)

acc_gbdt2 = cross_val_score(gbdt, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(acc_gbdt2,4))

gbdt_disp = plot_roc_curve(gbdt, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
gbdt_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')

##### SVM (Linear) #####
time_start = process_time()   

svm_lin = SVC(kernel='linear').fit(x_train, y_train)
svm_lin_pred = svm_lin.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, svm_lin_pred)
rep_svm_lin = classification_report(y_test, svm_lin_pred, target_names=target_names)
acc_svm_lin = accuracy_score(y_test, svm_lin_pred)

print("\n9. Linear SVM:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_svm_lin)

svm_lin_pred2 = cross_val_score(svm_lin, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(svm_lin_pred2,4))

svm_lin_disp = plot_roc_curve(svm_lin, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
svm_lin_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')

##### SVM (Gaussion Kernel (rbf)) #####
time_start = process_time()   

svm_rbf = SVC(kernel='rbf').fit(x_train, y_train)
svm_rbf_pred = svm_rbf.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, svm_rbf_pred)
rep_svm_rbf = classification_report(y_test, svm_rbf_pred, target_names=target_names)
acc_svm_rbf = accuracy_score(y_test, svm_rbf_pred)

print("\n10. SVM (Gaussion Kernel (rbf):")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_svm_rbf)

svm_rbf_pred2 = cross_val_score(svm_rbf, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(svm_rbf_pred2,4))

svm_rbf_disp = plot_roc_curve(svm_rbf, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
svm_rbf_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')

##### Extreme Gradient Boosting #####
time_start = process_time()   

xgb_model = xgb.XGBClassifier(max_depth=7, num_class=2, learning_rate=0.1, n_estimators=60, silent=True, objective='multi:softmax').fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)
TP, FP, TN, FN = conf_cal(y_test, xgb_pred)
rep_xgb = classification_report(y_test, xgb_pred, target_names=target_names)
acc_xgb = accuracy_score(y_test, xgb_pred)

print("\n11. Extreme Gradient Boosting:")
print("\nConfusion Matrix:")
print('                  Actual')
print('                (+)   (-)')
print('predicted (+) [',TP,FP,']')
print('          (-) [',FN,TN,']')

print('\nClassification report:')
print(rep_xgb)

xgb2 = cross_val_score(xgb_model, x, y, cv=10).mean()
print('10-fold cross validation accuracy:',round(xgb2,4))

xgb_model_disp = plot_roc_curve(xgb_model, x_test, y_test)
print('\nROC curve:')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 
xgb_model_disp.figure_.suptitle("ROC curve")
plt.show()

time_stop = process_time()
print('\nRun time:',(time_stop - time_start),'seconds\n')


##### ROC, AUC ######

nb_disp = plot_roc_curve(nb_gnb, x_test, y_test)
svm_lin_disp = plot_roc_curve(svm_lin, x_test, y_test, ax=nb_disp.ax_)
lr_disp = plot_roc_curve(lr, x_test, y_test, ax=svm_lin_disp.ax_)
print('\nROC curve comparision1:')
lr_disp.figure_.suptitle("ROC curve comparision")
plt.show()

dtree_disp = plot_roc_curve(dtree, x_test, y_test)
rfc_disp = plot_roc_curve(rfc, x_test, y_test, ax=dtree_disp.ax_)
knn_disp = plot_roc_curve(knn, x_test, y_test, ax=rfc_disp.ax_)
knn_bagging_disp = plot_roc_curve(knn_bagging, x_test, y_test, ax=knn_disp.ax_)
print('\nROC curve comparision2:')
knn_bagging_disp.figure_.suptitle("ROC curve comparision")
plt.show()

adb_disp = plot_roc_curve(adb, x_test, y_test)
gbdt_disp = plot_roc_curve(gbdt, x_test, y_test, ax=adb_disp.ax_)
svm_rbf_disp = plot_roc_curve(svm_rbf, x_test, y_test, ax=gbdt_disp.ax_)
xgb_model_disp = plot_roc_curve(xgb_model, x_test, y_test, ax=svm_rbf_disp.ax_)
print('\nROC curve comparision3:')
xgb_model_disp.figure_.suptitle("ROC curve comparision")
plt.show()
