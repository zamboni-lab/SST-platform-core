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

    line = 'a3b4c2e10b1'

    letters = []
    numbers = []

    for i in range(len(line)-1):

        if line[i].isalpha():
            letters.append(line[i])
            j = i+1  # next symbol is 100% a digit
            number = ""
            while line[j].isdigit():
                # while digit, collect symbols
                number += line[j]
                # watch out for the length
                if j+1 == len(line):
                    break
                else:
                    j += 1
            numbers.append(int(number))

    result = ''
    for i in range(len(letters)):
        result += letters[i] * numbers[i]

    print(result)








    pass
