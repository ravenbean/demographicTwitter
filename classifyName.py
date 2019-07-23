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
import pandas as pd
import sqlite3
import sys, os, pickle


def loadData(path):
    dt = pd.read_csv(path, encoding = 'utf-8-sig')
    dt = dt.dropna(how='all')

    map = {"Laki-Laki":1, "Perempuan":0}
    dt["jenis_kelamin"] = dt["jenis_kelamin"].map(map)
    featureColName = ["nama"]
    className = ["jenis_kelamin"]

    X = dt[featureColName].values
    y = dt[className].values
    print(X)
    print(y)
    return(X,y)

def getUserList(con3):
    cur = con3.execute("SELECT name FROM userscol ORDER BY name")
    alist = cur.fetchall()
    result = np.array(alist)
    return result

if os.path.isfile('./data/pipe_nb.pkl') is None:
    file_nb = open('./data/pipe_nb.pkl', 'rb')
    classifier = pickle.load(file_nb)
else:
    file_nb = open('./data/pipe_nb.pkl', 'wb')
    feat, label = loadData('./data/data-pemilih-kpu.csv')

    con3 = sqlite3.connect("C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/trainingSet.db")
    con3.execute("BEGIN")
    index = 0
    for row in feat:
        gender = 'Laki-Laki'
        if label[index][0] == 0:
            gender = 'Perempuan'
        print(row[0] +' : '+gender)
        split_name = row[0].split()
        first_name = split_name[0].replace("'","''")
        middle_name = ""
        last_name = ""
        if len(split_name) > 2:
            middle_name = split_name[1].replace("'","''")
            last_name = split_name[2].replace("'","''")
        elif len(split_name) > 1:
            last_name = split_name[1].replace("'","''")
        print("INSERT INTO data_pemilih_kpu_2017 VALUES ('"+row[0].replace("'","''")+"', '"+gender+"','"+first_name+"','"+middle_name+"','"+last_name+"')")
        con3.execute("INSERT OR IGNORE INTO data_pemilih_kpu_2017 VALUES ('"+row[0].replace("'","''")+"', '"+gender+"','"+first_name+"','"+middle_name+"','"+last_name+"')")
        con3.commit()
        index = index + 1

    X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.3, random_state=0)
    classifier = Pipeline([
        ('vectorizer', CountVectorizer( ngram_range=(1,1))),
        # ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
        # ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
        # ('classifier', LogisticRegression())
    ])
    classifier.fit(X_train.ravel(), y_train.ravel())
    pickle.dump(classifier, file_nb)



print(classifier.score(X_test.ravel(), y_test.ravel()))
# print(predicted)
# print(X_test)
# print(X_test.ravel())

# con3 = sqlite3.connect("D:/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres02.db")
# con3.execute("BEGIN")
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


# #PREDICT
# predicted = classifier.predict(userList.ravel())
# index = 0
# for row in predicted:
#     if row == 1:
#         jk = "Laki-Laki"
#     elif row == 0:
#         jk = "Perempuan"
#     query = "UPDATE userscol SET jenis_kelamin = '"+ jk +"' WHERE name LIKE '"+ str(userList.ravel()[index]) + "'"
#     index = index+1
#     print(query)
#     con3.execute(query)
#     con3.commit()