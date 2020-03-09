# path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/test_database.db"
# db_connector.create_qc_database(path)

import sqlite3

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

    # path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/qc_database.sqlite"
    #
    # conn = create_connection(path)
    # data, colnames = select_all_tasks(conn, "qc_meta")
    #
    # print()

    from pyteomics import mzxml

    data = list(mzxml.read("/Volumes/biol_imsb_sauer_1/users/Andrei/from Michelle/all of it/20200108_QC/mzXML/20190522_4GHz_Reid_001.mzXML"))
    # data = list(mzxml.read("/Volumes/biol_imsb_sauer_1/users/Andrei/from Michelle/all of it/20191108_QC/mzXML/20190522_4GHz_Reid_001.mzXML"))

    pass


