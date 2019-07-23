import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer,OneHotEncoder, Imputer, StandardScaler
import pandas as pd
import sqlite3
import sys, os, pickle, time
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from sklearn.model_selection import KFold, StratifiedKFold

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]

class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    print (data.shape)
    print (np.transpose(np.matrix(data)).shape)
    return np.transpose(np.matrix(data))

def loadData(con3):
    resTweet = []
    featureB = []   #bio
    featureC = []   #name
    featureD = []   #name+bio
    featureE = []   #tweet+bio+name
    featureF = []   #name+full_male+full_female
    featureI = []   #first_name + middle_name + last_name
    f_first_name = []   #first_name
    first_full_male = []
    first_full_female = []
    first_male = []
    first_female = []
    middle_male = []
    middle_female = []
    last_male = []
    last_female = []
    # cur = con3.execute("SELECT id, id_str, name, username, jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    cur = con3.execute("SELECT UC.id, UC.id_str, UC.name, UC.username, UC.jenis_kelamin, US.bio, US.location, US.likes, US.media, first_name, middle_name, last_name, IFNULL(first_male, 0) first_male, IFNULL(middle_male, 0) middle_male, IFNULL(last_male, 0) last_male, IFNULL(first_female, 0) first_female, IFNULL(middle_female, 0) middle_female, IFNULL(last_female, 0) last_female, IFNULL(first_full_male, 0) first_full_male, IFNULL(first_full_female, 0) first_full_female, US.join_date FROM userscol AS UC, users AS US WHERE UC.id = US.id ORDER BY UC.name")
    for row in cur:
        # query = "SELECT tweet FROM tweets WHERE user_id = "+str(row[0])+" LIMIT 200"
        # cur2 = con3.execute(query)
        combined_tweet = ''
        # for row2 in cur2:
        #     combined_tweet = combined_tweet + " " + str(row2[0])
        resTweet.append(combined_tweet)
        featureB.append(row[5])
        featureC.append(row[2]+" "+row[5]+" "+row[20])
        featureD.append(row[2]+" "+row[5])
        featureE.append(row[2]+" "+row[5]+" "+combined_tweet)
        #row[9] -> first_name
        featureF2d = []
        featureF2d.append(row[18])
        featureF2d.append(row[19])
        featureF2d.append(row[2])
        featureF2d.append(row[5]) #BIO
        featureF2d.append(row[20]) #JOIN_DATE
        featureF.append(featureF2d)
        f_first_name.append(row[9])
        first_full_male.append(row[18])
        first_full_female.append(row[19])
        #row[12] -> first_male
        first_male.append(row[12])
        middle_male.append(row[13])
        last_male.append(row[14])
        first_female.append(row[15])
        middle_female.append(row[16])
        last_female.append(row[17])
        featureI.append(row[9] + " " + row[10] + " " + row[11])
    resTweetAr = np.array(resTweet)
    print("Total Training: " + str(resTweetAr.__len__()))

    #training data jenis kelamin
    cur3 = con3.execute("SELECT jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    alist = cur3.fetchall()
    resLabel_jk = np.array(alist)
    print("Total Training (JK): " + str(resLabel_jk.__len__()))

    #training data umur
    # cur4 = con3.execute("SELECT kategori_umur2 FROM userscol WHERE kategori_umur2 IS NOT NULL ORDER BY name")
    # cur4 = con3.execute("SELECT CASE WHEN umur <= 38 THEN '<= 38' WHEN umur > 38 THEN '> 38' END AS kategori_umur3 FROM userscol WHERE umur IS NOT NULL ORDER BY name")
    cur4 = con3.execute("SELECT CASE WHEN umur > 24 AND umur < 40 THEN '25-39' WHEN umur >= 40 THEN '>= 40' WHEN umur <= 24 THEN '<= 24' END AS kategori_umur3 FROM userscol WHERE umur IS NOT NULL ORDER BY name")
    
    alist2 = cur4.fetchall()
    resLabel_umur = np.array(alist2)
    print("Total Training (kategori_umur): " + str(resLabel_umur.__len__()))

    #training data pekerjaan
    cur5 = con3.execute("SELECT CASE WHEN pekerjaan == '' THEN 'UNKNOWN' ELSE pekerjaan END pekerjaan FROM userscol WHERE pekerjaan IS NOT NULL ORDER BY name")
    alist3 = cur5.fetchall()
    resLabel_pekerjaan = np.array(alist3)
    print("Total Training (Pekerjaan): " + str(resLabel_pekerjaan.__len__()))

    return (resTweetAr, np.array(featureB), np.array(featureC), np.array(featureD), np.array(featureE), np.array(featureF), np.array(f_first_name), resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female, np.array(featureI))

def loadCaseData(con3):
    resTweet = []
    featureB = []   #bio
    featureC = []   #name
    featureD = []   #name+bio
    featureE = []   #tweet+bio+name
    featureF = []   #name+full_male+full_female
    f_first_name = []   #first_name
    first_full_male = []
    first_full_female = []
    first_male = []
    first_female = []
    middle_male = []
    middle_female = []
    last_male = []
    last_female = []
    # cur = con3.execute("SELECT id, id_str, name, username, jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    cur = con3.execute("SELECT UC.id, UC.id_str, UC.name, UC.username, UC.jenis_kelamin, US.bio, US.location, US.likes, US.media, first_name, middle_name, last_name, IFNULL(first_male, 0) first_male, IFNULL(middle_male, 0) middle_male, IFNULL(last_male, 0) last_male, IFNULL(first_female, 0) first_female, IFNULL(middle_female, 0) middle_female, IFNULL(last_female, 0) last_female, IFNULL(first_full_male, 0) first_full_male, IFNULL(first_full_female, 0) first_full_female, US.join_date FROM base.userscol AS UC, base.users AS US WHERE UC.id = US.id ORDER BY UC.name")
    for row in cur:
        # query = "SELECT tweet FROM tweets WHERE user_id = "+str(row[0])+" LIMIT 200"
        # cur2 = con3.execute(query)
        combined_tweet = ''
        # for row2 in cur2:
        #     combined_tweet = combined_tweet + " " + str(row2[0])
        resTweet.append(combined_tweet)
        featureB.append(row[5])
        featureC.append(row[2])
        featureD.append(row[2]+" "+row[5])
        featureE.append(row[2]+" "+row[5]+" "+combined_tweet)
        #row[9] -> first_name
        featureF2d = []
        featureF2d.append(row[18])
        featureF2d.append(row[19])
        featureF2d.append(row[2])
        featureF.append(featureF2d)
        f_first_name.append(row[9])
        first_full_male.append(row[18])
        first_full_female.append(row[19])
        #row[12] -> first_male
        first_male.append(row[12])
        middle_male.append(row[13])
        last_male.append(row[14])
        first_female.append(row[15])
        middle_female.append(row[16])
        last_female.append(row[17])
    resTweetAr = np.array(resTweet)
    print("Total Case: " + str(resTweetAr.__len__()))

    return (resTweetAr, np.array(featureB), np.array(featureC), np.array(featureD), np.array(featureE), np.array(featureF), np.array(f_first_name), resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

con3 = sqlite3.connect("trainingSet.db")
con3.execute("ATTACH 'feb2019_capres01_usertw.db' as usertw")
con3.execute("ATTACH 'feb2019_capres01.db' as base")
con3.execute("BEGIN")

feat, featB, featC, featD, featE, featF, f_first_name, resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female, featI = loadData(con3)

# print(featB)
print("Total feat B: " + str(featB.__len__()))
print("Total feat C: " + str(featC.__len__()))
print("Total feat D: " + str(featD.__len__()))
print("Total feat E: " + str(featE.__len__()))
print("Total Label JK: " + str(resLabel_jk.__len__()))
print("Total Label kategori_umur: " + str(resLabel_umur.__len__()))
print("Total f_first_name: " + str(f_first_name.__len__()))
# print(featB)


vector = CountVectorizer(ngram_range=(1,1))
x_temp = vector.fit_transform(featC)

vector = CountVectorizer(ngram_range=(1,1))
x_temp_G = vector.fit_transform(f_first_name)

x_matrix = []
x_matrix_G = []
x_matrix_H = []

index = 0
for f in x_temp.toarray():
    f = np.concatenate((f, np.array([first_full_male[index]]), np.array([first_full_female[index]])))
    x_matrix.append(f.tolist())
    index = index + 1
x_matrix = sparse.csr_matrix(x_matrix)
print(x_matrix.toarray())

index = 0
for f in x_temp_G.toarray():
    f = np.concatenate((f, np.array([first_full_male[index]]), np.array([first_full_female[index]])))
    x_matrix_G.append(f.tolist())
    index = index + 1
x_matrix_G = sparse.csr_matrix(x_matrix_G)
print(x_matrix_G.toarray())

index = 0
for f in x_temp.toarray():
    f = np.concatenate((f, np.array([first_male[index]]), np.array([middle_male[index]]), np.array([last_male[index]]), np.array([first_female[index]]), np.array([middle_female[index]]), np.array([last_female[index]]), np.array([first_full_male[index]]), np.array([first_full_female[index]])))
    x_matrix_H.append(f.tolist())
    index = index + 1
x_matrix_H = sparse.csr_matrix(x_matrix_H)
print(x_matrix_H.toarray())

#LABEL
Y_LABEL = resLabel_umur

#FEATURE F -> FULL NAME + FULL MALE + FULL FEMALE =============================================================================
avg_acc_mnb = 0
avg_acc_rf = 0
avg_acc_lr = 0
avg_acc_svm = 0

#MULTINOMIAL NB
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix[train], x_matrix[test], Y_LABEL[train], Y_LABEL[test]
    clf = MultinomialNB()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nMULTINOMIAL NAIVE BAYES - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_mnb = avg_acc_mnb + accuracy_score(y_test, predictions)

#RANDOM FOREST
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix[train], x_matrix[test], Y_LABEL[train], Y_LABEL[test]
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nRANDOM FOREST - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_rf = avg_acc_rf + accuracy_score(y_test, predictions)

#LOGISTIC REGRESSION
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix[train], x_matrix[test], Y_LABEL[train], Y_LABEL[test]
    clf = LogisticRegression()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nLOGISTIC REGRESSION - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_lr = avg_acc_lr + accuracy_score(y_test, predictions)

#SVM
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix[train], x_matrix[test], Y_LABEL[train], Y_LABEL[test]
    clf = LinearSVC()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nSVM - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_svm = avg_acc_svm + accuracy_score(y_test, predictions)

index = index - 1
F_avg_acc_mnb = avg_acc_mnb/index
F_avg_acc_rf = avg_acc_rf/index
F_avg_acc_lr = avg_acc_lr/index
F_avg_acc_svm = avg_acc_svm/index
print("FEATURE F")
print("Average accuracy MNB: " + str(avg_acc_mnb/index))
print("Average accuracy RF: " + str(avg_acc_rf/index))
print("Average accuracy LR: " + str(avg_acc_lr/index))
print("Average accuracy SVM: " + str(avg_acc_svm/index))

#====================================================================================================================






#FEATURE G -> FIRST NAME + FULL MALE + FULL FEMALE =============================================================================
avg_acc_mnb = 0
avg_acc_rf = 0
avg_acc_lr = 0
avg_acc_svm = 0

#MULTINOMIAL NB
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_G, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_G[train], x_matrix_G[test], Y_LABEL[train], Y_LABEL[test]
    clf = MultinomialNB()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nMULTINOMIAL NAIVE BAYES - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_mnb = avg_acc_mnb + accuracy_score(y_test, predictions)

#RANDOM FOREST
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_G, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_G[train], x_matrix_G[test], Y_LABEL[train], Y_LABEL[test]
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nRANDOM FOREST - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_rf = avg_acc_rf + accuracy_score(y_test, predictions)

#LOGISTIC REGRESSION
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_G, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_G[train], x_matrix_G[test], Y_LABEL[train], Y_LABEL[test]
    clf = LogisticRegression()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nLOGISTIC REGRESSION - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_lr = avg_acc_lr + accuracy_score(y_test, predictions)

#SVM
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_G, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_G[train], x_matrix_G[test], Y_LABEL[train], Y_LABEL[test]
    clf = LinearSVC()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nSVM - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_svm = avg_acc_svm + accuracy_score(y_test, predictions)

index = index - 1
print("FEATURE G")
G_avg_acc_mnb = avg_acc_mnb/index
G_avg_acc_rf = avg_acc_rf/index
G_avg_acc_lr = avg_acc_lr/index
G_avg_acc_svm = avg_acc_svm/index
print("Average accuracy MNB: " + str(avg_acc_mnb/index))
print("Average accuracy RF: " + str(avg_acc_rf/index))
print("Average accuracy LR: " + str(avg_acc_lr/index))
print("Average accuracy SVM: " + str(avg_acc_svm/index))

#====================================================================================================================






#FEATURE H -> name + first_male + middle_male + last_male + first_female + middle_female + last_female =============================================================================
avg_acc_mnb = 0
avg_acc_rf = 0
avg_acc_lr = 0
avg_acc_svm = 0

#MULTINOMIAL NB
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_H, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_H[train], x_matrix_H[test], Y_LABEL[train], Y_LABEL[test]
    clf = MultinomialNB()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nMULTINOMIAL NAIVE BAYES - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_mnb = avg_acc_mnb + accuracy_score(y_test, predictions)

#RANDOM FOREST
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_H, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_H[train], x_matrix_H[test], Y_LABEL[train], Y_LABEL[test]
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nRANDOM FOREST - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_rf = avg_acc_rf + accuracy_score(y_test, predictions)

#LOGISTIC REGRESSION
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_H, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_H[train], x_matrix_H[test], Y_LABEL[train], Y_LABEL[test]
    clf = LogisticRegression()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nLOGISTIC REGRESSION - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_lr = avg_acc_lr + accuracy_score(y_test, predictions)

#SVM
index = 1
# kf = KFold(n_splits=5)
kf = StratifiedKFold(n_splits=5)
for train, test in kf.split(x_matrix_H, Y_LABEL):
    X_train, X_test, y_train, y_test = x_matrix_H[train], x_matrix_H[test], Y_LABEL[train], Y_LABEL[test]
    clf = LinearSVC()
    clf.fit(X_train, y_train.ravel())
    predictions = clf.predict(X_test)
    print('\nSVM - '+ str(index) +':')
    print('Accuracy : ',accuracy_score(y_test, predictions))
    print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
    print('Classification report:\n',classification_report(y_test, predictions))
    index = index + 1
    avg_acc_svm = avg_acc_svm + accuracy_score(y_test, predictions)

index = index - 1
H_avg_acc_mnb = avg_acc_mnb/index
H_avg_acc_rf = avg_acc_rf/index
H_avg_acc_lr = avg_acc_lr/index
H_avg_acc_svm = avg_acc_svm/index

print("FEATURE F")
print("Average accuracy MNB: " + str(F_avg_acc_mnb))
print("Average accuracy RF: " + str(F_avg_acc_rf))
print("Average accuracy LR: " + str(F_avg_acc_lr))
print("Average accuracy SVM: " + str(F_avg_acc_svm))

print("FEATURE G")
print("Average accuracy MNB: " + str(G_avg_acc_mnb))
print("Average accuracy RF: " + str(G_avg_acc_rf))
print("Average accuracy LR: " + str(G_avg_acc_lr))
print("Average accuracy SVM: " + str(G_avg_acc_svm))

print("FEATURE H")
print("Average accuracy MNB: " + str(H_avg_acc_mnb))
print("Average accuracy RF: " + str(H_avg_acc_rf))
print("Average accuracy LR: " + str(H_avg_acc_lr))
print("Average accuracy SVM: " + str(H_avg_acc_svm))

#====================================================================================================================





