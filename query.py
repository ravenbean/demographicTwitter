import sqlite3
import re

con3 = sqlite3.connect("C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/trainingSet.db")
con3.execute("ATTACH 'feb2019_capres01_usertw.db' as usertw_01")
con3.execute("ATTACH 'feb2019_capres01.db' as base_01")
con3.execute("ATTACH 'feb2019_capres01_profile.db' as profile_01")
con3.execute("ATTACH 'feb2019_capres02_usertw.db' as usertw_02")
con3.execute("ATTACH 'feb2019_capres02.db' as base_02")
con3.execute("ATTACH 'feb2019_capres02_profile.db' as profile_02")
con3.execute("BEGIN")


# temp = con3.execute("SELECT COUNT(*) FROM base_01.userscol AS UC, base_01.user_profile AS US, usertw_01.all_tweet_user AS at WHERE UC.id = US.id AND UC.id = at.id ORDER BY UC.name")
# result = temp.fetchall()
# print(result)

# temp = con3.execute("SELECT COUNT(*) FROM base_01.tweets")
# result = temp.fetchall()
# print("Total tweet 01 all: " + str(result[0][0]))

# temp = con3.execute("SELECT * FROM base_01.tweets GROUP BY user_id")
# result = temp.fetchall()
# print("Total user 01 all: " + str(len(result)))

# temp = con3.execute("SELECT COUNT(*) FROM base_01.tweets tw, profile_01.users us WHERE tw.user_id = us.id")
# result = temp.fetchall()
# print("Total tweet 01 available: " + str(result[0][0]))

# temp = con3.execute("SELECT COUNT(*) FROM profile_01.users")
# result = temp.fetchall()
# print("Total user 01 available: " + str(result[0][0]))

# temp = con3.execute("SELECT COUNT(*) FROM base_02.tweets")
# result = temp.fetchall()
# print("\nTotal tweet 02 all: " + str(result[0][0]))

# temp = con3.execute("SELECT COUNT(*) FROM base_02.tweets GROUP BY user_id")
# result = temp.fetchall()
# print("Total user 02 all: " + str(len(result)))

# temp = con3.execute("SELECT COUNT(*) FROM base_02.tweets tw, profile_02.users us WHERE tw.user_id = us.id")
# result = temp.fetchall()
# print("Total tweet 02 available: " + str(result[0][0]))

# temp = con3.execute("SELECT COUNT(*) FROM profile_02.users")
# result = temp.fetchall()
# print("Total user 02 available: " + str(result[0][0]))

# index = 0
# for row in con3.execute("SELECT name, id FROM userscol WHERE first_name IS NULL"):
#     # print(row[2])
#     name = row[0].upper().replace("'","''").replace(".", " ").replace("@", " ").replace(":", " ").replace("IG ", " ").replace(" IG", " ")
#     name = re.sub(r'[^\w]', ' ', name)
#     split_name = name.strip().split()
#     # first_name = split_name[0]
#     # first_name = re.sub(r'[^\w]', ' ', first_name)
#     first_name = ""
#     middle_name = ""
#     last_name = ""
#     if len(split_name) > 0:
#         first_name = split_name[0]
#         first_name = re.sub(r'[^\w]', ' ', first_name)
#     if len(split_name) > 2:
#         middle_name = split_name[1]
#         middle_name = re.sub(r'[^\w]', ' ', middle_name)
#         last_name = split_name[2]
#         last_name = re.sub(r'[^\w]', ' ', last_name)
#     elif len(split_name) > 1:
#         last_name = split_name[1]
#         last_name = re.sub(r'[^\w]', ' ', last_name)
#     print(row[0] + " => " + first_name + " = " + middle_name + " = " + last_name)
#     query = "UPDATE userscol SET first_name = '" + first_name + "', middle_name = '"+ middle_name + "', last_name = '" + last_name + "' WHERE id = " + str(row[1])
#     print(query)
#     con3.execute(query)
# con3.commit()

for row in con3.execute("SELECT uc.id, uc.first_name, COUNT(*) as first_male FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.first_name LIKE dt.first_name AND dt.gender = 'Laki-Laki' GROUP BY uc.id"):
    query = "UPDATE userscol SET first_male = " + str(row[2]) + " WHERE id = " + str(row[0])
    print(query)
    con3.execute(query)

for row in con3.execute("SELECT uc.id, uc.middle_name, COUNT(*) as middle_male FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.middle_name LIKE dt.middle_name AND dt.gender = 'Laki-Laki' AND uc.middle_name != '' GROUP BY uc.id"):
    query = "UPDATE userscol SET middle_male = " + str(row[2]) + " WHERE id = " + str(row[0])
    print(query)
    con3.execute(query)
    
for row in con3.execute("SELECT uc.id, uc.last_name, COUNT(*) as last_male FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.last_name LIKE dt.last_name AND dt.gender = 'Laki-Laki' AND uc.last_name != '' GROUP BY uc.id"):
    query = "UPDATE userscol SET last_male = " + str(row[2]) + " WHERE id = " + str(row[0])
    print(query)
    con3.execute(query)

    
for row in con3.execute("SELECT uc.id, uc.first_name, COUNT(*) as first_male FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.first_name LIKE dt.first_name AND dt.gender = 'Perempuan' GROUP BY uc.id"):
    query = "UPDATE userscol SET first_female = " + str(row[2]) + " WHERE id = " + str(row[0])
    con3.execute(query)

for row in con3.execute("SELECT uc.id, uc.middle_name, COUNT(*) as middle_male FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.middle_name LIKE dt.middle_name AND dt.gender = 'Perempuan' AND uc.middle_name != '' GROUP BY uc.id"):
    query = "UPDATE userscol SET middle_female = " + str(row[2]) + " WHERE id = " + str(row[0])
    con3.execute(query)
    
for row in con3.execute("SELECT uc.id, uc.last_name, COUNT(*) as last_male FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.last_name LIKE dt.last_name AND dt.gender = 'Perempuan' AND uc.last_name != '' GROUP BY uc.id"):
    query = "UPDATE userscol SET last_female = " + str(row[2]) + " WHERE id = " + str(row[0])
    con3.execute(query)


    
for row in con3.execute("SELECT uc.id, uc.first_name, COUNT(*) as first_full_male FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.first_name LIKE dt.first_name OR uc.first_name LIKE dt.middle_name OR uc.first_name LIKE dt.last_name AND dt.gender = 'Laki-Laki' GROUP BY uc.id"):
    query = "UPDATE userscol SET first_full_male = " + str(row[2]) + " WHERE id = " + str(row[0])
    con3.execute(query)
    
for row in con3.execute("SELECT uc.id, uc.first_name, COUNT(*) as first_full_female FROM userscol uc, data_pemilih_kpu_2017 dt WHERE uc.first_name LIKE dt.first_name OR uc.first_name LIKE dt.middle_name OR uc.first_name LIKE dt.last_name AND dt.gender = 'Perempuan' GROUP BY uc.id"):
    query = "UPDATE userscol SET first_full_female = " + str(row[2]) + " WHERE id = " + str(row[0])
    con3.execute(query)

con3.commit()