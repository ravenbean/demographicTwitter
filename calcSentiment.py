from sentistrength_id import sentistrength
import sqlite3

con3 = sqlite3.connect("D:/KULIAH/KARYA AKHIR/BARU/SOLUTION/twint-master/SEARCHHASHTAG/HASIL/jokowi2periode.db")

config = dict()
config["negation"] = True
config["booster"]  = True
config["ungkapan"]  = True
config["consecutive"]  = True
config["repeated"]  = True
config["emoticon"]  = True
config["question"]  = True
config["exclamation"]  = True
config["punctuation"]  = True
senti = sentistrength(config)
# result = senti.main("agnezmo pintar dan cantik sekali tetapi lintah darat :)")
# print (senti.main("agnezmo pintar dan cantik sekali tetapi lintah darat :)"))
# print(result["kelas"])
# print(result["max_positive"])

index = 0
for row in con3.execute("SELECT * FROM tweets WHERE senti_positive IS NULL"):
    # print(row[2])
    result = senti.main(row[2])
    print(result)
    query = "UPDATE tweets SET senti_positive="+str(result["max_positive"])+", senti_negative="+str(result["max_negative"])+", senti_label='"+result["kelas"]+"' WHERE id="+str(row[0])
    # print(query)
    con3.execute(query)
    if index == 1000:
        con3.commit()
        index = 0
    index = index+1
con3.commit()

# for row in con3.execute("SELECT * FROM tweets WHERE senti_positive IS NOT NULL"):