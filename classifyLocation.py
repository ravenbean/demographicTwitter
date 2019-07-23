import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import sqlite3
import sys, os, pickle, time
from sklearn.base import BaseEstimator, TransformerMixin
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

def loadDataLocation(con3):
    featureA = []
    featureB = []
    featureC = []
    featureD = []
    featureE = []
    featureF = []
    index = 0
    # cur = con3.execute("SELECT name, username, bio, location_test, tweets, likes, media FROM users WHERE area IS NOT NULL ORDER BY name")
    cur = con3.execute("SELECT US.name, US.username, US.bio, US.location, US.tweets, US.likes, US.media, US.id FROM users US, loctw.all_tweet_user AT WHERE US.id = AT.id AND US.area IS NOT NULL ORDER BY US.name")
    for row in cur:
        featureA.append(row[2])
        featureB.append(row[0] + ' ' + row[2])
        featureC.append(row[2] + ' ' + row[3])
        query = "SELECT tweet, location, place FROM loctw.tweets WHERE user_id = "+str(row[7])+" LIMIT 30"
        # print(query)
        cur2 = con3.execute(query)
        combined_tweet = ''
        combined_tw_loc = ''
        for row2 in cur2:
            combined_tweet = combined_tweet + " " + str(row2[0])
            combined_tw_loc = combined_tw_loc + " " + str(row2[1]) + " " + str(row2[0]) + " " + str(row2[2])
        featureD.append(combined_tweet)
        # featureE.append(combined_tw_loc)
        featureE.append(row[0] + ' ' + row[2] + ' ' + combined_tweet)
        featureF.append(row[0] + ' ' + row[1] + ' ' + row[2] + ' ' + combined_tw_loc)
        index = index + 1
        print("index: " + str(index))

    # cur3 = con3.execute("SELECT US.area FROM users_byApjii US, loctw.all_tweet_user AT WHERE US.id = AT.id AND US.area IS NOT NULL ORDER BY US.name")
    # cur3 = con3.execute("SELECT CASE WHEN US.area LIKE 'Jawa%' OR US.area LIKE 'Jabodetabek' THEN 'Jawa' WHEN US.area LIKE 'Sulawesi' OR US.area LIKE 'Maluku' OR US.area LIKE 'Bali, NTB & NTT' THEN 'Indonesia Timur' ELSE US.area END area FROM users_byApjii US, loctw.all_tweet_user AT WHERE US.id = AT.id AND US.area IS NOT NULL ORDER BY US.name")
    cur3 = con3.execute("SELECT CASE WHEN area = 'Jabodetabek' OR area = 'Jawa Barat' OR area = 'Jawa Tengah & DIY' OR area = 'Jawa Timur' THEN 'JAWA' WHEN area != 'MULTIPLE' THEN 'NON JAWA' END area FROM users US, loctw.all_tweet_user AT WHERE US.id = AT.id AND US.area IS NOT NULL ORDER BY US.name")
    alist = cur3.fetchall()
    label_area = np.array(alist)

    npFeatD = np.array(featureD)
    print(len(npFeatD))

    return (np.array(featureA), np.array(featureB), np.array(featureC), npFeatD, np.array(featureE), np.array(featureF), label_area)

def loadData(con3):
    resTweet = []
    featureB = []   #bio
    featureC = []   #name
    featureD = []   #name+bio
    featureE = []   #tweet+bio+name
    featureF = []   #name+full_male+full_female
    first_full_male = []
    first_full_female = []
    # cur = con3.execute("SELECT id, id_str, name, username, jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    cur = con3.execute("SELECT UC.id, UC.id_str, UC.name, UC.username, UC.jenis_kelamin, US.bio, US.location, US.likes, US.media, first_name, middle_name, last_name, IFNULL(first_male, 0) first_male, IFNULL(middle_male, 0) middle_male, IFNULL(last_male, 0) last_male, IFNULL(first_female, 0) first_female, IFNULL(middle_female, 0) middle_female, IFNULL(last_female, 0) last_female, IFNULL(first_full_male, 0) first_full_male, IFNULL(first_full_female, 0) first_full_female FROM userscol AS UC, users AS US WHERE UC.id = US.id ORDER BY UC.name")
    for row in cur:
        query = "SELECT tweet FROM tweets WHERE user_id = "+str(row[0])+" LIMIT 200"
        cur2 = con3.execute(query)
        combined_tweet = ''
        for row2 in cur2:
            combined_tweet = combined_tweet + " " + str(row2[0])
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
        first_full_male.append(row[18])
        first_full_female.append(row[19])
    resTweetAr = np.array(resTweet)
    print("Total Training: " + str(resTweetAr.__len__()))

    #training data jenis kelamin
    cur3 = con3.execute("SELECT jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    alist = cur3.fetchall()
    resLabel_jk = np.array(alist)
    print("Total Training (JK): " + str(resLabel_jk.__len__()))

    #training data umur
    # cur4 = con3.execute("SELECT kategori_umur2 FROM userscol WHERE kategori_umur2 IS NOT NULL AND umur > 23 ORDER BY name")
    cur4 = con3.execute("SELECT CASE WHEN umur <= 26 THEN '23-26' WHEN umur >= 27 AND umur <= 30 THEN '27-30' WHEN umur > 30 THEN '> 30' END AS kategori_umur3 FROM userscol WHERE umur IS NOT NULL ORDER BY name")
    
    alist2 = cur4.fetchall()
    resLabel_umur = np.array(alist2)
    print("Total Training (kategori_umur2): " + str(resLabel_umur.__len__()))

    #training data pekerjaan
    cur5 = con3.execute("SELECT CASE WHEN pekerjaan == '' THEN 'UNKNOWN' ELSE pekerjaan END pekerjaan FROM userscol WHERE pekerjaan IS NOT NULL ORDER BY name")
    alist3 = cur5.fetchall()
    resLabel_pekerjaan = np.array(alist3)
    print("Total Training (Pekerjaan): " + str(resLabel_pekerjaan.__len__()))

    return (resTweetAr, np.array(featureB), np.array(featureC), np.array(featureD), np.array(featureE), np.array(featureF), resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female)

def loadCaseData(con3):
    resTweet = []
    resUserId = []
    # cur = con3.execute("SELECT id, id_str, name, username, umur FROM usertw.userscol ORDER BY name LIMIT 5000 OFFSET 4000")
    cur = con3.execute("SELECT id, id_str, name, username, umur FROM usertw.userscol WHERE umur IS null ORDER BY name LIMIT 4000")
    for row in cur:
        print("User: "+ str(row[2]))
        query = "SELECT tweet FROM usertw.tweets WHERE user_id = "+str(row[0])+" LIMIT 200"
        cur2 = con3.execute(query)
        combined_tweet = ''
        for row2 in cur2:
            combined_tweet = combined_tweet + " " + str(row2[0])
        resTweet.append(combined_tweet)
        resUserId.append(row[0])
    resTweetAr = np.array(resTweet)
    print(resTweetAr.__len__())
    return(resTweetAr, resUserId)

def getUserList(con3):
    cur = con3.execute("SELECT name FROM userscol ORDER BY name")
    alist = cur.fetchall()
    result = np.array(alist)
    return result

def getClassifier(con3):
    if os.path.isfile('./data/tweet_pipe_nb.pkl') is None:
        file_nb = open('./data/tweet_pipe_nb.pkl', 'rb')
        classifier = pickle.load(file_nb)
    else:
        file_nb = open('./data/tweet_pipe_nb.pkl', 'wb')
        feat, featB, featC, featD, featE, resLabel_jk, resLabel_umur, resLabel_pekerjaan = loadData(con3)
        # X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.25, random_state=33)
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="char_wb", ngram_range=(2,6))),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
        classifier.fit(feat.ravel(), resLabel_jk.ravel())
        pickle.dump(classifier, file_nb)
    return classifier

def runClassifier(feature, label, classifier, title):
    index = 1
    total_accuracy = 0
    kf = KFold(n_splits=5)
    # kf = StratifiedKFold(n_splits=5)
    for train, test in kf.split(feature, label):
        X_train, X_test, y_train, y_test = feature[train], feature[test], label[train], label[test]
        # print(X_train)
        model = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1,1))),
            # ('vectorizer', TfidfVectorizer(lowercase=False, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
            # ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        # print(title+' - '+str(index)+':')
        # print('Accuracy : ',accuracy_score(y_test, predictions))
        print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
        print('Classification report:\n',classification_report(y_test, predictions))
        total_accuracy = total_accuracy + accuracy_score(y_test, predictions)
        index = index + 1
    index = index - 1
    # print("Average accuracy: " + str(total_accuracy/index))
    # print("\n")
    return total_accuracy/index

def runClassifierTFIDF(feature, label, classifier, title):
    index = 1
    total_accuracy = 0
    kf = KFold(n_splits=5)
    # kf = StratifiedKFold(n_splits=5)
    for train, test in kf.split(feature, label):
        X_train, X_test, y_train, y_test = feature[train], feature[test], label[train], label[test]
        # print(X_train)
        model = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1,1))),
            # ('vectorizer', TfidfVectorizer(lowercase=False, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        # print(title+' - '+str(index)+':')
        # print('Accuracy : ',accuracy_score(y_test, predictions))
        print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
        print('Classification report:\n',classification_report(y_test, predictions))
        total_accuracy = total_accuracy + accuracy_score(y_test, predictions)
        index = index + 1
    index = index - 1
    # print("Average accuracy: " + str(total_accuracy/index))
    # print("\n")
    return total_accuracy/index

con3 = sqlite3.connect("user_location.db")
# con3 = sqlite3.connect("C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/twint-master/SEARCHHASHTAG/user_location.db")
con3.execute("ATTACH 'tweet_user_location.db' as loctw")
con3.execute("ATTACH 'feb2019_capres02_usertw.db' as usertw")
con3.execute("ATTACH 'feb2019_capres02.db' as base")
con3.execute("BEGIN")

# feat, featB, featC, featD, featE, featF, resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female = loadData(con3)
featA, featB, featC, featD, featE, featF, label_area = loadDataLocation(con3)

# LOAD REAL CASE DATA        ##################################################

# start = time.time()
# print("start")
# caseData, caseUser = loadCaseData(con3)
# finishload = time.time()
# # print(caseData)
# print("finish load data: "+ str(finishload -  start))

# END OF LOAD REAL CASE DATA ##################################################

# TEST CLASSIFY USING TRAINING & TEST DATA     ######################################################


# print(featB)

# print(featB)



# print("\nJENIS KELAMIN =======================================================")
# Y_LABEL = resLabel_jk
# print("\nFEATURE A =======================================================")
# A_MNB = runClassifier(feat, Y_LABEL, MultinomialNB(), "MNB")
# A_RF = runClassifier(feat, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
# A_LR = runClassifier(feat, Y_LABEL, LogisticRegression(), "LR")
# A_SVM = runClassifier(feat, Y_LABEL, LinearSVC(), "SVM")

# print("\nFEATURE B =======================================================")
# B_MNB = runClassifier(featB, Y_LABEL, MultinomialNB(), "MNB")
# B_RF = runClassifier(featB, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
# B_LR = runClassifier(featB, Y_LABEL, LogisticRegression(), "LR")
# B_SVM = runClassifier(featB, Y_LABEL, LinearSVC(), "SVM")

# print("\nFEATURE C =======================================================")
# C_MNB = runClassifier(featC, Y_LABEL, MultinomialNB(), "MNB")
# C_RF = runClassifier(featC, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
# C_LR = runClassifier(featC, Y_LABEL, LogisticRegression(), "LR")
# C_SVM = runClassifier(featC, Y_LABEL, LinearSVC(), "SVM")

# print("\nFEATURE D =======================================================")
# D_MNB = runClassifier(featD, Y_LABEL, MultinomialNB(), "MNB")
# D_RF = runClassifier(featD, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
# D_LR = runClassifier(featD, Y_LABEL, LogisticRegression(), "LR")
# D_SVM = runClassifier(featD, Y_LABEL, LinearSVC(), "SVM")

# print("\nFEATURE E =======================================================")
# E_MNB = runClassifier(featE, Y_LABEL, MultinomialNB(), "MNB")
# E_RF = runClassifier(featE, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
# E_LR = runClassifier(featE, Y_LABEL, LogisticRegression(), "LR")
# E_SVM = runClassifier(featE, Y_LABEL, LinearSVC(), "SVM")

# print("A_MNB: " + str(A_MNB))
# print("A_RF: " + str(A_RF))
# print("A_LR: " + str(A_LR))
# print("A_SVM: " + str(A_SVM))
# print("\n")
# print("B_MNB: " + str(B_MNB))
# print("B_RF: " + str(B_RF))
# print("B_LR: " + str(B_LR))
# print("B_SVM: " + str(B_SVM))
# print("\n")
# print("C_MNB: " + str(C_MNB))
# print("C_RF: " + str(C_RF))
# print("C_LR: " + str(C_LR))
# print("C_SVM: " + str(C_SVM))
# print("\n")
# print("D_MNB: " + str(D_MNB))
# print("D_RF: " + str(D_RF))
# print("D_LR: " + str(D_LR))
# print("D_SVM: " + str(D_SVM))
# print("\n")
# print("E_MNB: " + str(E_MNB))
# print("E_RF: " + str(E_RF))
# print("E_LR: " + str(E_LR))
# print("E_SVM: " + str(E_SVM))


print("\nUMUR =======================================================")
Y_LABEL = label_area
print("\nFEATURE A =======================================================")
A_MNB = runClassifier(featA, Y_LABEL, MultinomialNB(), "MNB")
A_RF = runClassifier(featA, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
A_LR = runClassifier(featA, Y_LABEL, LogisticRegression(), "LR")
A_SVM = runClassifier(featA, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE B =======================================================")
B_MNB = runClassifier(featB, Y_LABEL, MultinomialNB(), "MNB")
B_RF = runClassifier(featB, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
B_LR = runClassifier(featB, Y_LABEL, LogisticRegression(), "LR")
B_SVM = runClassifier(featB, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE C =======================================================")
C_MNB = runClassifier(featC, Y_LABEL, MultinomialNB(), "MNB")
C_RF = runClassifier(featC, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
C_LR = runClassifier(featC, Y_LABEL, LogisticRegression(), "LR")
C_SVM = runClassifier(featC, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE D =======================================================")
D_MNB = runClassifier(featD, Y_LABEL, MultinomialNB(), "MNB")
D_RF = runClassifier(featD, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
D_LR = runClassifier(featD, Y_LABEL, LogisticRegression(), "LR")
D_SVM = runClassifier(featD, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE E =======================================================")
E_MNB = runClassifier(featE, Y_LABEL, MultinomialNB(), "MNB")
E_RF = runClassifier(featE, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
E_LR = runClassifier(featE, Y_LABEL, LogisticRegression(), "LR")
E_SVM = runClassifier(featE, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE F =======================================================")
F_MNB = runClassifier(featF, Y_LABEL, MultinomialNB(), "MNB")
F_RF = runClassifier(featF, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
F_LR = runClassifier(featF, Y_LABEL, LogisticRegression(), "LR")
F_SVM = runClassifier(featF, Y_LABEL, LinearSVC(), "SVM")


print("\nUSING TF-IDF =======================================================")
print("\nFEATURE A =======================================================")
A_MNB_TFIDF = runClassifierTFIDF(featA, Y_LABEL, MultinomialNB(), "MNB")
A_RF_TFIDF = runClassifierTFIDF(featA, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
A_LR_TFIDF = runClassifierTFIDF(featA, Y_LABEL, LogisticRegression(), "LR")
A_SVM_TFIDF = runClassifierTFIDF(featA, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE B =======================================================")
B_MNB_TFIDF = runClassifierTFIDF(featB, Y_LABEL, MultinomialNB(), "MNB")
B_RF_TFIDF = runClassifierTFIDF(featB, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
B_LR_TFIDF = runClassifierTFIDF(featB, Y_LABEL, LogisticRegression(), "LR")
B_SVM_TFIDF = runClassifierTFIDF(featB, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE C =======================================================")
C_MNB_TFIDF = runClassifierTFIDF(featC, Y_LABEL, MultinomialNB(), "MNB")
C_RF_TFIDF = runClassifierTFIDF(featC, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
C_LR_TFIDF = runClassifierTFIDF(featC, Y_LABEL, LogisticRegression(), "LR")
C_SVM_TFIDF = runClassifierTFIDF(featC, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE D =======================================================")
D_MNB_TFIDF = runClassifierTFIDF(featD, Y_LABEL, MultinomialNB(), "MNB")
D_RF_TFIDF = runClassifierTFIDF(featD, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
D_LR_TFIDF = runClassifierTFIDF(featD, Y_LABEL, LogisticRegression(), "LR")
D_SVM_TFIDF = runClassifierTFIDF(featD, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE E =======================================================")
E_MNB_TFIDF = runClassifierTFIDF(featE, Y_LABEL, MultinomialNB(), "MNB")
E_RF_TFIDF = runClassifierTFIDF(featE, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
E_LR_TFIDF = runClassifierTFIDF(featE, Y_LABEL, LogisticRegression(), "LR")
E_SVM_TFIDF = runClassifierTFIDF(featE, Y_LABEL, LinearSVC(), "SVM")

print("\nFEATURE F =======================================================")
F_MNB_TFIDF = runClassifierTFIDF(featF, Y_LABEL, MultinomialNB(), "MNB")
F_RF_TFIDF = runClassifierTFIDF(featF, Y_LABEL, RandomForestClassifier(n_estimators=10, n_jobs=-1), "RF")
F_LR_TFIDF = runClassifierTFIDF(featF, Y_LABEL, LogisticRegression(), "LR")
F_SVM_TFIDF = runClassifierTFIDF(featF, Y_LABEL, LinearSVC(), "SVM")

print("Total feat A: " + str(featA.__len__()))
print("Total feat B: " + str(featB.__len__()))
print("Total feat C: " + str(featC.__len__()))
print("Total feat D: " + str(featD.__len__()))
print("Total feat E: " + str(featE.__len__()))
print("Total feat F: " + str(featF.__len__()))
# print("Total feat F: " + str(featF.__len__()))
print("Total Label Area: " + str(label_area.__len__()))
print("\n")

print("A_MNB: " + str(A_MNB) + " <> A_MNB_TFIDF: " + str(A_MNB_TFIDF))
print("A_RF: " + str(A_RF) + " <> A_RF_TFIDF: " + str(A_RF_TFIDF))
print("A_LR: " + str(A_LR) + " <> A_LR_TFIDF: " + str(A_LR_TFIDF))
print("A_SVM: " + str(A_SVM) + " <> A_SVM_TFIDF: " + str(A_SVM_TFIDF))
print("\n")
print("B_MNB: " + str(B_MNB) + " <> B_MNB_TFIDF: " + str(B_MNB_TFIDF))
print("B_RF: " + str(B_RF) + " <> B_RF_TFIDF: " + str(B_RF_TFIDF))
print("B_LR: " + str(B_LR) + " <> B_LR_TFIDF: " + str(B_LR_TFIDF))
print("B_SVM: " + str(B_SVM) + " <> B_SVM_TFIDF: " + str(B_SVM_TFIDF))
print("\n")
print("C_MNB: " + str(C_MNB) + " <> C_MNB_TFIDF: " + str(C_MNB_TFIDF))
print("C_RF: " + str(C_RF) + " <> C_RF_TFIDF: " + str(C_RF_TFIDF))
print("C_LR: " + str(C_LR) + " <> C_LR_TFIDF: " + str(C_LR_TFIDF))
print("C_SVM: " + str(C_SVM) + " <> C_SVM_TFIDF: " + str(C_SVM_TFIDF))
print("\n")
print("D_MNB: " + str(D_MNB) + " <> D_MNB_TFIDF: " + str(D_MNB_TFIDF))
print("D_RF: " + str(D_RF) + " <> D_RF_TFIDF: " + str(D_RF_TFIDF))
print("D_LR: " + str(D_LR) + " <> D_LR_TFIDF: " + str(D_LR_TFIDF))
print("D_SVM: " + str(D_SVM) + " <> D_SVM_TFIDF: " + str(D_SVM_TFIDF))
print("\n")
print("E_MNB: " + str(E_MNB) + " <> E_MNB_TFIDF: " + str(E_MNB_TFIDF))
print("E_RF: " + str(E_RF) + " <> E_RF_TFIDF: " + str(E_RF_TFIDF))
print("E_LR: " + str(E_LR) + " <> E_LR_TFIDF: " + str(E_LR_TFIDF))
print("E_SVM: " + str(E_SVM) + " <> E_SVM_TFIDF: " + str(E_SVM_TFIDF))
print("\n")
print("F_MNB: " + str(F_MNB) + " <> F_MNB_TFIDF: " + str(F_MNB_TFIDF))
print("F_RF: " + str(F_RF) + " <> F_RF_TFIDF: " + str(F_RF_TFIDF))
print("F_LR: " + str(F_LR) + " <> F_LR_TFIDF: " + str(F_LR_TFIDF))
print("F_SVM: " + str(F_SVM) + " <> F_SVM_TFIDF: " + str(F_SVM_TFIDF))


#MULTINOMIAL NAIVE BAYES /////////////////////////////////////////////////////////////////////////////////////////////////////
# X_train, X_test, y_train, y_test = train_test_split(featC, resLabel_jk, test_size=0.3, random_state=0)
# classifier = Pipeline([
#     # ('vectorizer', CountVectorizer(ngram_range=(1,1))),
#     # ('vectorizer', TfidfVectorizer(lowercase=False, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
#     # ('tfidf', TfidfTransformer()),
#     ('classifier', MultinomialNB())
#     # ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
#     # ('classifier', LogisticRegression())
#     # ('classifier', LinearSVC())
# ])


# classifier.fit(X_train.ravel(), y_train.ravel())
# predictions = classifier.predict(X_test.ravel())
# print('Accuracy : ', classifier.score(X_test.ravel(), y_test.ravel()))
# print('\nMULTINOMIAL NAIVE BAYES:')
# print('Accuracy : ',accuracy_score(y_test, predictions))
# print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
# print('Classification report:\n',classification_report(y_test, predictions))


# #RANDOM FOREST /////////////////////////////////////////////////////////////////////////////////////////////////////
# X_train, X_test, y_train, y_test = train_test_split(featD, resLabel_jk, test_size=0.3, random_state=0)
# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(ngram_range=(1,1))),
#     # ('vectorizer', TfidfVectorizer(lowercase=False, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
#     # ('tfidf', TfidfTransformer()),
#     ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
#     # ('classifier', LogisticRegression())
#     # ('classifier', LinearSVC())
# ])
# classifier.fit(X_train.ravel(), y_train.ravel())
# predictions = classifier.predict(X_test.ravel())
# print('\nRANDOM FOREST:')
# print('Accuracy : ',accuracy_score(y_test, predictions))
# print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
# print('Classification report:\n',classification_report(y_test, predictions))


# #LOGISTIC REGRESSION /////////////////////////////////////////////////////////////////////////////////////////////////////
# X_train, X_test, y_train, y_test = train_test_split(featD, resLabel_jk, test_size=0.3, random_state=0)
# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(ngram_range=(1,1))),
#     # ('vectorizer', TfidfVectorizer(lowercase=False, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
#     # ('tfidf', TfidfTransformer()),
#     ('classifier', LogisticRegression())
#     # ('classifier', LinearSVC())
# ])
# classifier.fit(X_train.ravel(), y_train.ravel())
# predictions = classifier.predict(X_test.ravel())
# print('\nLOGISTIC REGRESSION:')
# print('Accuracy : ',accuracy_score(y_test, predictions))
# print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
# print('Classification report:\n',classification_report(y_test, predictions))


# #SVM /////////////////////////////////////////////////////////////////////////////////////////////////////
# X_train, X_test, y_train, y_test = train_test_split(featD, resLabel_jk, test_size=0.3, random_state=0)
# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(ngram_range=(1,1))),
#     # ('vectorizer', TfidfVectorizer(lowercase=False, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
#     # ('tfidf', TfidfTransformer()),
#     ('classifier', LinearSVC())
# ])
# classifier.fit(X_train.ravel(), y_train.ravel())
# predictions = classifier.predict(X_test.ravel())
# print('\nSVM:')
# print('Accuracy : ',accuracy_score(y_test, predictions))
# print('Confusion matrix:\n',confusion_matrix(y_test, predictions))
# print('Classification report:\n',classification_report(y_test, predictions))










# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(max_features=1500, min_df=5, max_df=0.7)),
#     ('tfidf', TfidfTransformer()),
#     ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
# ])
# classifier.fit(X_train.ravel(), y_train.ravel())
# print(classifier.score(X_test.ravel(), y_test.ravel()))

# END OFTEST CLASSIFY USING TRAINING & TEST DATA ######################################################


# print(predicted)
# print(X_test)
# print(X_test.ravel())

# con3 = sqlite3.connect("D:/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/trainingSet.db")
# con3.execute("BEGIN")
# loadData(con3)
# userList = getUserList(con3)
# # print(result)

# index = 0
# predicted = classifier.predict(X_test.ravel())
# for row in predicted:
#     if row == 1:
#         jk = "Laki-Laki"
#     elif row == 0:
#         jk = "Perempuan"
#     query = "UPDATE userscol SET jenis_kelamin = '"+ jk +"' WHERE name LIKE '"+ str(userList.ravel()[index]) + "'"
#     index = index+1
#     print(query)


#PREDICT USING REAL CASE ##################################################################
# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(max_features=1500, min_df=5, max_df=0.7)),
#     ('tfidf', TfidfTransformer()),
#     ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
# ])
# classifier.fit(X_train.ravel(), y_train.ravel())

# predicted = classifier.predict(caseData.ravel())
# index = 0
# for row in predicted:
#     query = "UPDATE base.userscol SET umur = '"+ row +"' WHERE id_str LIKE '"+ str(caseUser[index]) + "'"
#     index = index+1
#     print(query)
#     con3.execute(query)
#     con3.commit()