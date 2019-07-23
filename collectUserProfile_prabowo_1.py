
import sqlite3
import sys
sys.path.insert(0, 'C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/twint-master')
# import Twint
import twint

con3 = sqlite3.connect("C:/Users/taufa/Documents/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres02_usertw.db")
con3.execute("BEGIN")

c = twint.Config()
# c.Store_csv = True
c.Count = True


c.Database = "feb2019_capres02_profile.db"
c.Location = True
# c.Limit = 400
# c.User_full = True
# c.Profile_full = True
c.User_info = True

cur = con3.execute("SELECT id, username FROM userscol WHERE id NOT IN (SELECT id FROM user_profile) ORDER BY id")
for row in cur:
    print("\nCollecting: " + str(row[1]))
    c.Username = str(row[1])
    twint.run.Lookup(c)