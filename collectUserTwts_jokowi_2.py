import twint
import sqlite3
import sys

def getUserTweets():
    con3 = sqlite3.connect("D:/KULIAH/KARYA AKHIR/BARU/SOLUTION/demographic_twitter/feb2019_capres01_cpy.db")
    con3.execute("BEGIN")
    result = con3.execute("SELECT user_id, user_id_str, name, screen_name FROM tweets WHERE user_id NOT IN (SELECT id FROM userscol) GROUP BY user_id ORDER BY name LIMIT 4000 OFFSET 4000")
    for row in result:
        try:
            print(row[0])
            query = "INSERT INTO userscol (id, id_str, name, username) VALUES ("+ str(row[0]) + ", '"+ row[1] + "', '"+ row[2] + "', '"+ row[3] + "')"
            print(query)
            con3.execute(query)
            con3.commit()
            search(row[0])
        except Exception as e:
            print(e)
            pass

def search(user):
    c = twint.Config()
    c.Count = True
    c.Database = "feb2019_capres01_usertw_cpy.db"
    c.User_id = user
    c.Limit = 200
    # c.Hide_output = True
    c.Format = "Tweet id: {id} | Date: {date}| Time: {time}| Username: {username} | hashtag: {hashtags}"
    twint.run.Profile(c)



getUserTweets()
