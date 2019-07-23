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

def loadDataLocation(con3):
    featureA = []
    featureB = []
    featureC = []
    featureD = []
    featureE = []
    featureF = []
    index = 0
    # cur = con3.execute("SELECT name, username, bio, location_test, tweets, likes, media FROM users WHERE area IS NOT NULL ORDER BY name")
    cur = con3.execute("SELECT US.name, US.username, US.bio, US.location, US.tweets, US.likes, US.media, US.id FROM users_byApjii US, loctw.all_tweet_user AT WHERE US.id = AT.id AND US.area IS NOT NULL ORDER BY US.name")
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
    cur3 = con3.execute("SELECT CASE WHEN area = 'Jabodetabek' OR area = 'Jawa Barat' OR area = 'Jawa Tengah & DIY' OR area = 'Jawa Timur' THEN 'JAWA' WHEN area != 'MULTIPLE' THEN 'NON JAWA' END area FROM users_byApjii US, loctw.all_tweet_user AT WHERE US.id = AT.id AND US.area IS NOT NULL ORDER BY US.name")
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
        query = "SELECT tweet FROM tweets WHERE user_id = "+str(row[0])+" LIMIT 20"
        cur2 = con3.execute(query)
        combined_tweet = ''
        for row2 in cur2:
            combined_tweet = combined_tweet + " " + str(row2[0])
        resTweet.append(combined_tweet)
        featureB.append(row[5])
        featureC.append(row[2])
        featureD.append(row[2]+" "+row[5] + " " + row[20])
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
    # cur4 = con3.execute("SELECT CASE WHEN umur <= 30 THEN '<= 30' WHEN umur > 30 THEN '> 30' END AS kategori_umur3 FROM userscol WHERE umur IS NOT NULL ORDER BY name")
    cur4 = con3.execute("SELECT CASE WHEN umur > 24 AND umur < 40 THEN '25-39' WHEN umur >= 40 THEN '>= 40' WHEN umur <= 24 THEN '<= 24' END AS kategori_umur3 FROM userscol WHERE umur IS NOT NULL ORDER BY name")
    alist2 = cur4.fetchall()
    resLabel_umur = np.array(alist2)
    print("Total Training (kategori_umur): " + str(resLabel_umur.__len__()))

    #training data pekerjaan
    cur5 = con3.execute("SELECT CASE WHEN pekerjaan == '' THEN 'UNKNOWN' ELSE pekerjaan END pekerjaan FROM userscol WHERE pekerjaan IS NOT NULL ORDER BY name")
    alist3 = cur5.fetchall()
    resLabel_pekerjaan = np.array(alist3)
    print("Total Training (Pekerjaan): " + str(resLabel_pekerjaan.__len__()))

    return (resTweetAr, np.array(featureB), np.array(featureC), np.array(featureD), np.array(featureE), np.array(featureF), np.array(f_first_name), resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female)

def loadCaseData(con3):
    id_list = []
    featureA = [] #bio
    featureB = [] #name + bio
    featureC = [] #not used
    featureD = [] #tweet
    featureE = [] #name + bio + tweet
    featureF = [] #name + username + bio + tweet(+tweet loc)
    index = 0
    # cur = con3.execute("SELECT id, id_str, name, username, jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    cur = con3.execute("SELECT UC.id, UC.id_str, UC.name, UC.username, UC.jenis_kelamin, US.bio, US.location, US.likes, US.media FROM base.userscol AS UC, base.user_profile AS US WHERE UC.id = US.id ORDER BY UC.name")
    for row in cur:
        id_list.append(row[0])
        featureA.append(row[5])
        featureB.append(row[2] + ' ' + row[5])
        featureC.append(row[2] + ' ' + row[3])
        query = "SELECT tweet, location, place FROM tweets WHERE user_id = "+str(row[0])+" LIMIT 30"
        cur2 = con3.execute(query)
        combined_tweet = ''
        combined_tw_loc = ''
        for row2 in cur2:
            combined_tweet = combined_tweet + " " + str(row2[0])
            combined_tw_loc = combined_tw_loc + " " + str(row2[1]) + " " + str(row2[0]) + " " + str(row2[2])
        featureD.append(combined_tweet)
        # featureE.append(combined_tw_loc)
        featureE.append(row[2] + ' ' + row[5] + ' ' + combined_tweet)
        featureF.append(row[2] + ' ' + row[5] + ' ' + row[3] + ' ' + combined_tw_loc)
        
        print(index)
        index = index + 1

    return (id_list, np.array(featureA), np.array(featureB), np.array(featureC), np.array(featureD), np.array(featureE), np.array(featureF))


def runClassifier(train_feature, train_label, case_feature, case_name_feature, case_id_list, classifier, title):
    model = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1,1))),
        # ('vectorizer', TfidfVectorizer(lowercase=False, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
        ('tfidf', TfidfTransformer()),
        ('classifier', classifier)
    ])
    model.fit(train_feature, train_label.ravel())
    predictions = model.predict(case_feature)
    index_i = 0
    for row in predictions:
        print("PREDICT " + case_name_feature[index_i])
        query = "UPDATE base.userscol SET lokasi = '"+ row +"' WHERE id_str LIKE '"+ str(case_id_list[index_i]) + "'"
        # index = index+1
        print(query)
        con3.execute(query)
        index_i = index_i + 1
        con3.commit()



con3 = sqlite3.connect("user_location.db")
con3.execute("ATTACH 'tweet_user_location.db' as loctw")
con3.execute("ATTACH 'feb2019_capres01_usertw.db' as usertw")
con3.execute("ATTACH 'feb2019_capres01.db' as base")
con3.execute("BEGIN")

# feat, featB, featC, featD, featE, featF, f_first_name, resLabel_jk, resLabel_umur, resLabel_pekerjaan, first_full_male, first_full_female, first_male, middle_male, last_male, first_female, middle_female, last_female = loadData(con3)
featA, featB, featC, featD, featE, featF, label_area = loadDataLocation(con3)
case_id_list, case_featA, case_featB, case_featC, case_featD, case_featE, case_featF = loadCaseData(con3)

# print(featB)
print("Total feat B: " + str(featB.__len__()))
print("Total feat C: " + str(featC.__len__()))
print("Total feat D: " + str(featD.__len__()))
print("Total feat E: " + str(featE.__len__()))
# print("Total Label JK: " + str(resLabel_jk.__len__()))
# print("Total Label kategori_umur: " + str(resLabel_umur.__len__()))
# print("Total f_first_name: " + str(f_first_name.__len__()))
# print(featB)
print("\n")
# print(case_featC)


runClassifier(featF, label_area, case_featF, case_featC, case_id_list, LinearSVC(), "SVM")









