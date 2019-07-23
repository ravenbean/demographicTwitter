import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
import sqlite3
import sys, os, pickle, time


def loadData(con3):
    resTweet = []

    cur_tw = con3.execute("SELECT all_tweet FROM all_tweet_user ORDER BY name")
    allTweets = cur_tw.fetchall()
    resTweetAr = np.array(allTweets)
    print(resTweetAr.__len__())

    # cur = con3.execute("SELECT id, id_str, name, username, jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    # for row in cur:
    #     query = "SELECT tweet FROM tweets WHERE user_id = "+str(row[0])+" LIMIT 200"
    #     cur2 = con3.execute(query)
    #     combined_tweet = ''
    #     for row2 in cur2:
    #         combined_tweet = combined_tweet + " " + str(row2[0])
    #     resTweet.append(combined_tweet)
    # resTweetAr = np.array(resTweet)
    # print(resTweetAr.__len__())

    #training data jenis kelamin
    cur3 = con3.execute("SELECT jenis_kelamin FROM userscol WHERE jenis_kelamin IS NOT NULL ORDER BY name")
    alist = cur3.fetchall()
    resLabel_jk = np.array(alist)
    print(resLabel_jk.__len__())

    #training data umur
    cur4 = con3.execute("SELECT umur FROM userscol WHERE umur IS NOT NULL ORDER BY name")
    alist2 = cur4.fetchall()
    resLabel_umur = np.array(alist2)
    print(resLabel_umur.__len__())

    #training data pekerjaan
    cur5 = con3.execute("SELECT CASE WHEN pekerjaan == '' THEN 'UNKNOWN' ELSE pekerjaan END pekerjaan FROM userscol WHERE pekerjaan IS NOT NULL ORDER BY name")
    alist3 = cur5.fetchall()
    resLabel_pekerjaan = np.array(alist3)
    print(resLabel_pekerjaan.__len__())

    return (resTweetAr, resLabel_jk, resLabel_umur, resLabel_pekerjaan)

def loadCaseData(con3):
    resTweet = []
    resUserId = []
    # cur = con3.execute("SELECT id, id_str, name, username, umur FROM usertw.userscol ORDER BY name LIMIT 5000 OFFSET 4000")
    # cur = con3.execute("SELECT id, id_str, name, username, umur FROM usertw.userscol WHERE umur IS null ORDER BY name LIMIT 4000")
    # for row in cur:
    #     print("User: "+ str(row[2]))
    #     query = "SELECT tweet FROM usertw.tweets WHERE user_id = "+str(row[0])+" LIMIT 200"
    #     cur2 = con3.execute(query)
    #     combined_tweet = ''
    #     for row2 in cur2:
    #         combined_tweet = combined_tweet + " " + str(row2[0])
    #     resTweet.append(combined_tweet)
    #     resUserId.append(row[0])
    # resTweetAr = np.array(resTweet)
    # print(resTweetAr.__len__())

    cur_tw = con3.execute("SELECT all_tweet FROM usertw.all_tweet_user ORDER BY name")
    allTweets = cur_tw.fetchall()
    resTweetAr = np.array(allTweets)
    print(resTweetAr.__len__())

    cur_user = con3.execute("SELECT id_str FROM usertw.all_tweet_user ORDER BY name")
    allTweets_user = cur_user.fetchall()
    resUserId = np.array(allTweets_user)
    print(resUserId.__len__())

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
        feat, resLabel_jk, resLabel_umur, resLabel_pekerjaan = loadData(con3)
        # X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.25, random_state=33)
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="char_wb", ngram_range=(2,6))),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
        classifier.fit(feat.ravel(), resLabel_jk.ravel())
        pickle.dump(classifier, file_nb)
    return classifier

con3 = sqlite3.connect("D:/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/trainingSet.db")
con3.execute("ATTACH 'D:/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres02_usertw.db' as usertw")
con3.execute("ATTACH 'D:/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres02.db' as base")
con3.execute("BEGIN")

feat, resLabel_jk, resLabel_umur, resLabel_pekerjaan = loadData(con3)

# LOAD REAL CASE DATA        ##################################################

start = time.time()
print("start")
caseData, caseUser = loadCaseData(con3)
finishload = time.time()
# print(caseData)
print("finish load data: "+ str(finishload -  start))

# END OF LOAD REAL CASE DATA ##################################################

# TEST CLASSIFY USING TRAINING & TEST DATA     ######################################################

X_train, X_test, y_train, y_test = train_test_split(feat, resLabel_umur, test_size=0.25)
# classifier = Pipeline([
#     ('vectorizer', CountVectorizer(analyzer="char_wb", ngram_range=(2,6))),
#     ('tfidf', TfidfTransformer()),
#     ('classifier', MultinomialNB())
    # ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
    # ('classifier', LogisticRegression())
# ])
# classifier.fit(X_train.ravel(), y_train.ravel())
# print(classifier.score(X_test.ravel(), y_test.ravel()))

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
classifier = Pipeline([
    ('vectorizer', CountVectorizer(max_features=1500, min_df=5, max_df=0.7)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
])
classifier.fit(feat.ravel(), resLabel_umur.ravel())

print("START PREDICTING...")
predicted = classifier.predict(caseData.ravel())
index = 0
for row in predicted:
    query = "UPDATE base.userscol SET umur = '"+ row +"' WHERE id_str LIKE '"+ str(caseUser[index]).replace("['","").replace("']","") + "'"
    index = index+1
    print(query)
    con3.execute(query)
con3.commit()