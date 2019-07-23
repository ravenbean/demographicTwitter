from os import listdir
from os.path import isfile, join
import sqlite3


# print(onlyfiles)
con3 = sqlite3.connect("C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres02_usertw.db")
con3.execute("ATTACH 'C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres02_profile.db' as dba")
con3.execute("ATTACH 'C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres02.db' as base")
con3.execute("BEGIN")

combine = "INSERT OR IGNORE INTO base.user_profile SELECT * FROM dba.users"
print(combine)
con3.execute(combine)
con3.commit()
# combine = "INSERT OR IGNORE INTO base.userscol (id, id_str, name, username) SELECT * FROM userscol"
# print(combine)
# con3.execute(combine)
# con3.commit()
# con3.execute("detach database dba")
    


