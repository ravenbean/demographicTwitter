import twint
import sqlite3
import sys

def search(keyword):
    c = twint.Config()
    c.Count = True
    c.Database = "feb2019_capres02.db"
    c.Replies = False
    c.Retweets = False
    c.Lowercase = True
    c.Stats = False
#     c.Hide_output = True
    c.Format = "Tweet id: {id} | Date: {date}| Time: {time}| Username: {username} | hashtag: {hashtags}"

#     c.Search = keyword
#     c.Until = "2019-02-28"
#     c.Since = "2019-02-27"
#     c.Retries_count = 1000
#     twint.run.Search(c)
        

    for x in range(2, 29, 2):
        c.Search = keyword
        c.Until = "2019-02-"+ f'{x:02}'
        c.Since = "2019-02-"+ f'{x-1:02}'
        c.Retries_count = 1000
        twint.run.Search(c)


# search("#2019prabowosandi")
# search("#2019gantipresiden")
# search("#2019prabowopresiden")
# search("#indonesiamenang")

