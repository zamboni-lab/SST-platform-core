# path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/test_database.db"
# db_connector.create_qc_database(path)

import sqlite3, numpy, scipy

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(str(e))

    return conn


def select_all_tasks(conn, table_name):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + table_name)

    colnames = [description[0] for description in cur.description]

    return cur.fetchall(), colnames


if __name__ == "__main__":

    # a = [0, 2, 2, 0, 1, 3]
    # b = [1, 2, 0, 0, 0, 3]
    #
    # i = 0
    # while i < len(a):
    #     if a[i] == b[i] == 0:
    #         a.pop(i)
    #         b.pop(i)
    #     i += 1
    #
    # print(a)
    # print(b)
    pass






