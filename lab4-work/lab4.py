import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/") 
my_db = myclient["mydatabase"]

print(myclient.list_database_names())

dblist = myclient.list_database_names()
if "mydatabase" in dblist:
  print("The database exists")