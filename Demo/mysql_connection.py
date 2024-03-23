import pymysql

host = 'localhost'
port = 3306
db = 'vqa'
user = 'vqa_test'
password = '123456'

def connect():
    conn = pymysql.connect(host=host, port=port, db=db, user=user, password=password)
    return conn

## 测试代码
con=connect()
print(con)
