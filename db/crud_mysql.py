
# encoding: utf-8
'''
Author: Baoyun Peng
Date: 2022-03-13 23:37:14
LastEditTime: 2022-03-22 20:08:57
Description: CRUD functions for MySQL
 reference: https://www.cnblogs.com/xuanzhi201111/p/5144982.html

'''
import MySQLdb
from .table_schema import ai_model_center, ai_model_data_center, ai_model_finish_template_info, ai_model_template_module_info

# insert_sql = "INSERT INTO ai_model_center ( \
#     id, \
#     service_type, \
#     model_name, \
#     model_identification, \
#     model_unique_code, \
#     template_id, \
#     inspection_type, \
#     inspection_position, \
#     inspection_project, \
#     model_introduction, \
#     state, \
#     ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

def get_connect(user, passwd, db, host='127.0.0.1', port=3306, charset='utf8'):
    conn = MySQLdb.connect(
        host=host,
        port=port,
        user=user,
        passwd=passwd,
        db=db,
        charset=charset
    )
    cursor = conn.cursor()
    return conn, cursor

def close_connect(conn, cursor):
    cursor.close()
    conn.close()

def db_execute_val(conn, cursor, sql, val=None):
    if val is None:
        cursor.execute(sql)
    else:
        cursor.executemany(sql, val)
    conn.commit()
    if 'select' in sql.lower():
        # get the query result
        result = cursor.fetchall()
    elif 'insert' in sql.lower():
        # get the number of insert row 
        result = cursor.rowcount
    else:
        result = None
    return result

def gen_insert_sql(table_name, table_schema):
    _sql = f"INSERT INTO {table_name} ("
    _posix = ""
    for key in table_schema:
        _sql += key + ','
        if 'time' in key: 
            _posix += "'%s',"
        else:
             _posix += "%s,"
    _sql = _sql.strip(',') + ") VALUES ("
    _posix = _posix.strip(',') + ")"
    return _sql + _posix

def gen_delete_sql(table_name, conditions=None):
    if conditions is None:
        _sql = f"DELETE FROM {table_name} "
    else:
        _sql = f"DELETE FROM {table_name} WHERE {conditions}"
    return _sql

def gen_select_sql(table_name, query_conds=None):
    if query_conds is None:
        _sql = f"SELECT * FROM {table_name}"
    else:
        # select those data satisfying the query conditions
        _sql = f"SELECT * FROM {table_name} where {query_conds}"
    return _sql

def gen_update_sql(table_name, condition, new_value):
    _sql = f"UPDATE {table_name} SET {new_value} WHERE {condition}"
    return _sql

if __name__ == "__main__":
    conn, cursor = get_connect('download', 'Down@0221', 'ai_model_quality_control')
    select_sql = gen_select_sql('ai_model_data_center')
    result = db_execute_val(conn, cursor, select_sql)
    table_name = 'ai_model_data_center'
    condition_str = "id=20"
    new_value = "ai_score=87"
    _sql = gen_update_sql(table_name, condition_str, new_value)
    insert_res = db_execute_val(conn, cursor, _sql)
    close_connect(conn, cursor)
    import pdb
    pdb.set_trace()