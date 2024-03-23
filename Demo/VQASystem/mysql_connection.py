import pymysql

host = '114.55.42.184'
port = 3306
db = 'vqa'
user = 'root'
password = 'hc38827206'

host = 'localhost'
port = 3306
db = 'vqa'
user = 'vqa_test'
password = '123456'

def connect():
    conn = pymysql.connect(host=host, port=port, db=db, user=user, password=password)
    return conn

def test():
    print("ssdfaa")