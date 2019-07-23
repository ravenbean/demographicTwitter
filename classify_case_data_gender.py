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
from sklearn.model_selection import KFold

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
    cur = con3.execute("SELECT UC.id, UC.id_str, UC.name, UC.username, UC.jenis_kelamin, US.bio, US.location, US.likes, US.media, first_name, middle_name, last_name, IFNULL(first_male, 0) first_male, IFNULL(middle_male, 0) middle_male, IFNULL(last_male, 0) last_male, IFNULL(first_female, 0) first_female, IFNULL(middle_female, 0) middle_female, IFNULL(last_female, 0) last_female, IFNULL(first_full_male, 0) first_full_male, IFNULL(first_full_female, 0) first_full_female FROM userscol AS UC, users AS US WHERE UC.id = US.id ORDER BY UC.name")
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
    print("Total Training: " + str(resTweetAr.__len__()))

    #training data jenis kelamin
    cur3 = con3.execute("SELECT jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    alist = cur3.fetchall()
    resLabel_jk = np.array(alist)
    print("Total Training (JK): " + str(resLabel_jk.__len__()))

    #training data umur
    cur4 = con3.execute("SELECT kategori_umur2 FROM userscol WHERE kategori_umur2 IS NOT NULL ORDER BY name")
    alist2 = cur4.fetchall()
    resLabel_umur = np.array(alist2)
    print("Total Training (kategori_umur2): " + str(resLabel_umur.__len__()))

    #training data pekerjaan
    cur5 = con3.execute("SELECT CASE WHEN pekerjaan == '' THEN 'UNKNOWN' ELSE pekerjaan END pekerjaan FROM userscol WHERE pekerjaan IS NOT NULL ORDER BY name")
    alist3 = cur5.fetchall()
    resLabel_pekerjaan = np.array(alist3)
    print("Total Training (Pekerjaan): " + str(resLabel_pekerjaan.__len__()))

    return (resTweetAr, np.array(featureB), np.array(featureC), np.array(featureD), np.array(featureE), np.array(featureF), np.array(f_first_name), resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female)

def loadCaseData(con3):
    id_list = []
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
    index = 0
    # cur = con3.execute("SELECT id, id_str, name, username, jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    cur = con3.execute("SELECT UC.id, UC.id_str, UC.name, UC.username, UC.jenis_kelamin, US.bio, US.location, US.likes, US.media, first_name, middle_name, last_name, IFNULL(first_male, 0) first_male, IFNULL(middle_male, 0) middle_male, IFNULL(last_male, 0) last_male, IFNULL(first_female, 0) first_female, IFNULL(middle_female, 0) middle_female, IFNULL(last_female, 0) last_female, IFNULL(first_full_male, 0) first_full_male, IFNULL(first_full_female, 0) first_full_female FROM base.userscol AS UC, base.user_profile AS US WHERE UC.id = US.id ORDER BY UC.name")
    for row in cur:
        id_list.append(row[0])
        # query = "SELECT tweet FROM tweets WHERE user_id = "+str(row[0])+" LIMIT 20"
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
        print(index)
        index = index + 1
    resTweetAr = np.array(resTweet)
    print("Total Case: " + str(resTweetAr.__len__()))

    return (id_list, resTweetAr, np.array(featureB), np.array(featureC), np.array(featureD), np.array(featureE), np.array(featureF), np.array(f_first_name), first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female)


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
con3.execute("ATTACH 'feb2019_capres02_usertw.db' as usertw")
con3.execute("ATTACH 'feb2019_capres02.db' as base")
con3.execute("BEGIN")

feat, featB, featC, featD, featE, featF, f_first_name, resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female = loadData(con3)

case_id_list, case_feat, case_featB, case_featC, case_featD, case_featE, case_featF, case_f_first_name, case_first_full_male, case_first_full_female, case_first_male, case_middle_male, case_last_male, case_first_female, case_middle_female, case_last_female = loadCaseData(con3)

# print(featB)
print("Total feat B: " + str(featB.__len__()))
print("Total feat C: " + str(featC.__len__()))
print("Total feat D: " + str(featD.__len__()))
print("Total feat E: " + str(featE.__len__()))
print("Total Label JK: " + str(resLabel_jk.__len__()))
print("Total Label kategori_umur: " + str(resLabel_umur.__len__()))
print("Total f_first_name: " + str(f_first_name.__len__()))
# print(featB)
print("\n")
# print(case_featC)

vector = CountVectorizer(ngram_range=(1,1))
x_temp = vector.fit_transform(featC)

vector = CountVectorizer(ngram_range=(1,1))
x_case_temp = vector.fit_transform(case_featC)

vector = CountVectorizer(ngram_range=(1,1))
x_temp_G = vector.fit_transform(f_first_name)

x_matrix = []
x_matrix_G = []
x_matrix_H = []
x_matrix_Case = []

index = 0
for f in x_temp.toarray():
    f = np.concatenate((f, np.array([first_full_male[index]]), np.array([first_full_female[index]])))
    x_matrix.append(f.tolist())
    index = index + 1
x_matrix = sparse.csr_matrix(x_matrix)
# print(x_matrix.toarray())

index = 0
for f in x_temp_G.toarray():
    f = np.concatenate((f, np.array([first_full_male[index]]), np.array([first_full_female[index]])))
    x_matrix_G.append(f.tolist())
    index = index + 1
x_matrix_G = sparse.csr_matrix(x_matrix_G)
# print(x_matrix_G.toarray())

index = 0
for f in x_case_temp.toarray():
    f = np.concatenate((f, np.array([case_first_male[index]]), np.array([case_middle_male[index]]), np.array([case_last_male[index]]), np.array([case_first_female[index]]), np.array([case_middle_female[index]]), np.array([case_last_female[index]]), np.array([case_first_full_male[index]]), np.array([case_first_full_female[index]])))
    x_matrix_H.append(f.tolist())
    index = index + 1
x_matrix_H = sparse.csr_matrix(x_matrix_H)
# print(x_matrix_H.toarray())

index = 0
for f in x_temp.toarray():
    f = np.concatenate((f, np.array([first_male[index]]), np.array([middle_male[index]]), np.array([last_male[index]]), np.array([first_female[index]]), np.array([middle_female[index]]), np.array([last_female[index]]), np.array([first_full_male[index]]), np.array([first_full_female[index]])))
    x_matrix_Case.append(f.tolist())
    index = index + 1
x_matrix_Case = sparse.csr_matrix(x_matrix_Case)
# print(x_matrix_H.toarray())

#LABEL
Y_LABEL = resLabel_jk

#FEATURE F -> FULL NAME + FULL MALE + FULL FEMALE =============================================================================
avg_acc_mnb = 0
avg_acc_rf = 0
avg_acc_lr = 0
avg_acc_svm = 0


#FEATURE H -> name + first_male + middle_male + last_male + first_female + middle_female + last_female =============================================================================
avg_acc_mnb = 0
avg_acc_rf = 0
avg_acc_lr = 0
avg_acc_svm = 0

#MULTINOMIAL NB
# index = 1
# kf = KFold(n_splits=5)
# for train, test in kf.split(x_matrix_H, Y_LABEL):
#     X_train, X_test, y_train, y_test = x_matrix_H[train], x_matrix_H[test], Y_LABEL[train], Y_LABEL[test]
#     clf = MultinomialNB()
#     clf.fit(X_train, y_train.ravel())
#     predictions = clf.predict(X_test)
#     print('\nMULTINOMIAL NAIVE BAYES - '+ str(index) +':')
#     print('Accuracy : ',accuracy_score(y_test, predictions))
#     print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
#     print('Classification report:\n',classification_report(y_test, predictions))
#     index = index + 1
#     avg_acc_mnb = avg_acc_mnb + accuracy_score(y_test, predictions)

index_i = 0
for row in case_featC:
    fc = np.append(featC, row)
    newFc = np.array(fc)
    vector = CountVectorizer(ngram_range=(1,1))
    x_temp = vector.fit_transform(newFc)

    # print(first_full_male)
    t_fm = list(first_male)
    t_mm = list(middle_male)
    t_lm = list(last_male)
    t_ff = list(first_female)
    t_mf = list(middle_female)
    t_lf = list(last_female)
    temp_ffm = list(first_full_male)
    temp_fff = list(first_full_female)

    temp_ffm.append(case_first_full_male[index_i])
    temp_fff.append(case_first_full_female[index_i])
    t_fm.append(case_first_male[index_i])
    t_mm.append(case_middle_male[index_i])
    t_lm.append(case_last_male[index_i])
    t_ff.append(case_first_female[index_i])
    t_mf.append(case_middle_female[index_i])
    t_lf.append(case_last_female[index_i])

    # print("======================")
    # print(first_full_male)
    # print(temp_ffm)
    # print(temp_ffm)

    index = 0
    last_index = len(x_temp.toarray()) - 1
    x_matrix = []
    for f in x_temp.toarray():
        if index < last_index:
            f = np.concatenate((f, np.array([t_fm[index]]), np.array([t_mm[index]]), np.array([t_lm[index]]), np.array([t_ff[index]]), np.array([t_mf[index]]), np.array([t_lf[index]]), np.array([temp_ffm[index]]), np.array([temp_fff[index]])))
            # f = np.concatenate((f, np.array([temp_ffm[index]]), np.array([temp_fff[index]])))
            x_matrix.append(f.tolist())
            index = index + 1
    x_matrix = sparse.csr_matrix(x_matrix)
    # print(x_matrix.toarray())

    # print(len(x_matrix.toarray()))
    # print(len(x_temp.toarray()))
    # print(len(temp_ffm))
    # print(len(temp_fff))
    # print(last_index)
    # print(x_temp.toarray()[last_index])
    # print(np.array([temp_ffm[last_index]]))
    # print(np.array([temp_fff[last_index]]))
    x_case = []
    xx_temp = list(x_temp.toarray()[last_index])
    xx_temp.append(t_fm[last_index])
    xx_temp.append(t_mm[last_index])
    xx_temp.append(t_lm[last_index])
    xx_temp.append(t_ff[last_index])
    xx_temp.append(t_mf[last_index])
    xx_temp.append(t_lf[last_index])
    xx_temp.append(temp_ffm[last_index])
    xx_temp.append(temp_fff[last_index])
    # print(xx_temp)
    # xc_temp = np.concatenate(x_temp.toarray()[last_index], np.array([temp_ffm[last_index]]), np.array([temp_fff[last_index]]) )
    xc_temp = np.array(xx_temp)
    # print(xc_temp)
    x_case.append(xc_temp.tolist())
    x_case_matrix = sparse.csr_matrix(x_case)
    # print(len(x_case))
    # print(x_case_matrix)

    clf = MultinomialNB()
    # print(len(x_matrix.toarray()))
    # print(len(Y_LABEL))
    clf.fit(x_matrix, Y_LABEL.ravel())
    predictions = clf.predict(x_case_matrix)
    # index = 0
    for row in predictions:
        print("PREDICT " + case_featC[index_i])
        query = "UPDATE base.userscol SET jenis_kelamin = '"+ row +"' WHERE id_str LIKE '"+ str(case_id_list[index_i]) + "'"
        # index = index+1
        print(query)
        con3.execute(query)
        con3.commit()
    index_i = index_i + 1

#====================================================================================================================





