import pandas, datetime
import mysql.connector

from src import mysql_source
from src.constants import all_metrics
from src.qcmg import metrics_generator
from src.constants import get_buffer_id


def create_mysql_connection():
    """ Creates a database connection to the MySQL database specified by db_file. """

    db = None
    try:
        db = mysql.connector.connect(host=mysql_source.mysqlHost,
                                     user=mysql_source.mysqlUser,
                                     password=mysql_source.mysqlPW,
                                     database=mysql_source.mysqlDB)
        return db
    except Exception as e:
        print(e)

    return db


def create_table(db, create_table_sql):
    """ Copied from sqlite_connector.py.
        Creates a table from the create_table_sql statement. """
    try:
        c = db.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print("Error creating table:", e)


def fetch_table(conn, table_name):
    """ Copied from sqlite_connector.py.
        Gets data from the table_name given connection. """
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + table_name)
    colnames = [description[0] for description in cur.description]

    return cur.fetchall(), colnames


def update_column_in_database(conn, table_name, colname, col_values, rowname, row_values):
    """ Adapted from sqlite_connector.py.
        This method updates the whole quality column in qc_meta table of the given database.
        Called to update QC features and QC tunes databases,
        when qualities are newly generated for QC metrics database. """

    sql = ''' UPDATE table_name
              SET column_name = %s
              WHERE row_name = %s '''

    sql = sql.replace("table_name", table_name).replace("column_name", colname).replace("row_name", rowname)

    cur = conn.cursor()

    multiple_entries = [(col_values[i], row_values[i]) for i in range(len(row_values))]

    cur.executemany(sql, multiple_entries)
    conn.commit()


def add_column_to_database(conn, table_name, column_name, column_type):
    """ Copied from sqlite_connector.py.
        This method adds a new column of given name and type to an existing database. """

    sql = ''' ALTER TABLE table_name
              ADD COLUMN column_name column_type '''

    sql = sql.replace("table_name", table_name).replace("column_name", column_name).replace("column_type", column_type)

    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()


def remove_row_from_table_by_id(conn, table_name, id):
    """ Copied from sqlite_connector.py.
        This method removes a row of given id from an existing database. """

    sql = ''' DELETE FROM table_name
              WHERE id=value '''

    sql = sql.replace("table_name", table_name).replace("value", str(id))

    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()


def remove_row_from_all_databases_by_id(run_id):
    """ Adapted from sqlite_connector.py.
        This method removes a row with a given id from all tables of all MySQL databases. """

    conn = create_mysql_connection()
    remove_row_from_table_by_id(conn, 'qc_features_1', run_id)
    remove_row_from_table_by_id(conn, 'qc_features_2', run_id)
    remove_row_from_table_by_id(conn, 'qc_meta', run_id)

    conn = create_mysql_connection()
    remove_row_from_table_by_id(conn, 'qc_metrics', run_id)
    remove_row_from_table_by_id(conn, 'qc_metrics_qualities', run_id)
    remove_row_from_table_by_id(conn, 'qc_meta', run_id)

    conn = create_mysql_connection()
    remove_row_from_table_by_id(conn, 'qc_tunes', run_id)
    remove_row_from_table_by_id(conn, 'qc_meta', run_id)


def insert_qc_metrics(db, qc_run, meta_id):
    """ Adapted from sqlite_connector.py.
        Adds last runs QC  metrics values to the table. """

    qc_metrics = (
        meta_id,
        qc_run['acquisition_date'],
        qc_run['quality'],
        *qc_run['metrics_values']
    )

    # NOTE: in MySQL we changed 'signal' to 'sig'
    sql = ''' INSERT INTO qc_metrics(meta_id,acquisition_date,quality,resolution_200,resolution_700,
                                    average_accuracy,chemical_dirt,instrument_noise,isotopic_presence,
                                    transmission,fragmentation_305,fragmentation_712,baseline_25_150,
                                    baseline_50_150,baseline_25_650,baseline_50_650,sig,
                                    s2b,s2n)

                          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) '''

    cur = db.cursor()
    cur.execute(sql, qc_metrics)
    db.commit()

    return cur.lastrowid


def insert_qc_metrics_qualities(db, qc_run, meta_id):
    """ Adapted from sqlite_connector.py.
        Adds last run QC metrics qualities (0 or 1 per each metric) to the table. """

    qc_metrics = (
        meta_id,
        qc_run['acquisition_date'],
        qc_run['quality'],
        *qc_run['metrics_qualities']
    )

    sql = ''' INSERT INTO qc_metrics_qualities(meta_id,acquisition_date,quality,resolution_200,resolution_700,
                                    average_accuracy,chemical_dirt,instrument_noise,isotopic_presence,
                                    transmission,fragmentation_305,fragmentation_712,baseline_25_150,
                                    baseline_50_150,baseline_25_650,baseline_50_650,sig,
                                    s2b,s2n)

                          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) '''

    cur = db.cursor()
    cur.execute(sql, qc_metrics)
    db.commit()

    return cur.lastrowid


def insert_qc_features(db, new_qc_run, meta_id):
    """ Adapted from sqlite_connector.py.
        Adds last runs QC features values to two tables: features are splitted in two parts,
        since SQLite only has 2000 cols at max for a single table. """

    qc_features_1 = (
        meta_id,
        new_qc_run['acquisition_date'],
        new_qc_run['quality'],
        *new_qc_run['features_values'][:1996]
    )

    qc_features_2 = (
        meta_id,
        new_qc_run['acquisition_date'],
        new_qc_run['quality'],
        *new_qc_run['features_values'][1996:]
    )

    # first portion
    sql_1 = """ INSERT INTO qc_features_1(meta_id,acquisition_date,quality,"""
    sql_1 += ",".join(new_qc_run["features_names"][:1996]) + """ ) VALUES("""  # features names
    sql_1 += ",".join(["%s" for x in range(1999)]) + """) """  # 1999 question marks

    # second portion
    sql_2 = """ INSERT INTO qc_features_2(meta_id,acquisition_date,quality,"""
    sql_2 += ",".join(new_qc_run["features_names"][1996:]) + """ ) VALUES("""  # features names
    sql_2 += ",".join(["%s" for x in range(
        len(new_qc_run['features_values']) - 1996 + 3)]) + """) """  # question marks for the rest of values

    cur = db.cursor()
    cur.execute(sql_1, qc_features_1)
    cur.execute(sql_2, qc_features_2)
    db.commit()

    return cur.lastrowid


def insert_qc_tunes(db, new_qc_run, meta_id):
    """ Adapted from sqlite_connector.py.
        Adds last runs QC tunes to a table. """

    single_names_string = ",".join(new_qc_run['tunes_names'][0]) + ","
    single_names_string += ",".join(new_qc_run['tunes_names'][1]) + ","
    single_names_string += ",".join(new_qc_run['tunes_names'][2])
    single_names_string = single_names_string.replace(":", "_").replace(".", "_")\
        .replace("(", "_").replace(")","").replace("/", "_")

    single_values_list = new_qc_run["tunes_values"][0]
    single_values_list.extend(new_qc_run["tunes_values"][1])
    single_values_list.extend(new_qc_run["tunes_values"][2])

    qc_tunes = (
        meta_id,
        new_qc_run['acquisition_date'],
        new_qc_run['quality'],
        *single_values_list
    )

    sql = """ INSERT INTO qc_tunes(meta_id,acquisition_date,quality,"""
    sql += single_names_string + """ ) VALUES("""  # tunes names
    sql += ",".join(["%s" for x in range(len(single_values_list) + 3)]) + """) """  # question marks

    cur = db.cursor()
    cur.execute(sql, qc_tunes)
    db.commit()

    return cur.lastrowid


def insert_qc_meta(db, qc_run):
    """ Adapted from sqlite_connector.py.
        Adds last runs meta info to the table. """

    qc_meta = (
        qc_run['md5'],
        qc_run['processing_date'],
        qc_run['acquisition_date'],
        qc_run['instrument'],
        qc_run['quality'],
        qc_run['user'],
        qc_run['user_comment'],
        qc_run['chemical_mix_id'],
        qc_run['buffer_id'],
        qc_run['msfe_version'],
        qc_run['scans_processed']['normal'][0],
        qc_run['scans_processed']['normal'][1],
        qc_run['scans_processed']['normal'][2],
        qc_run['scans_processed']['chemical_noise'][0],
        qc_run['scans_processed']['instrument_noise'][0]
    )

    sql = ''' INSERT INTO qc_meta(md5,processing_date,acquisition_date,instr,quality,user,user_comment,
                                  chemical_mix_id,buffer_id,msfe_version,norm_scan_1,norm_scan_2,
                                  norm_scan_3,chem_scan_1,inst_scan_1)

                          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) '''

    cur = db.cursor()
    cur.execute(sql, qc_meta)
    db.commit()

    return cur.lastrowid


def create_qc_metrics_database():
    """ Adapted from sqlite_connector.py. """

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            md5 text,
                                            processing_date text,
                                            acquisition_date text,
                                            instr integer,
                                            quality integer,
                                            user text,
                                            user_comment text,
                                            chemical_mix_id text,
                                            buffer_id text,
                                            msfe_version text,
                                            norm_scan_1 integer,
                                            norm_scan_2 integer,
                                            norm_scan_3 integer,
                                            chem_scan_1 integer,
                                            inst_scan_1 integer
                                        ); """

    sql_create_qc_metrics_table = """ CREATE TABLE IF NOT EXISTS qc_metrics (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            meta_id integer,
                                            acquisition_date text,
                                            quality integer,
                                            resolution_200 integer,
                                            resolution_700 integer,
                                            average_accuracy real,
                                            chemical_dirt integer,
                                            instrument_noise integer,
                                            isotopic_presence real,
                                            transmission real,
                                            fragmentation_305 real,
                                            fragmentation_712 real,
                                            baseline_25_150 integer,
                                            baseline_50_150 integer,
                                            baseline_25_650 integer,
                                            baseline_50_650 integer,
                                            sig integer,
                                            s2b real,
                                            s2n real,
                                            constraint fk_meta foreign key (meta_id) 
                                                references qc_meta(id) 
                                                on delete cascade 
                                                on update cascade  
                                        ); """

    sql_create_qc_metrics_qualities_table = """ CREATE TABLE IF NOT EXISTS qc_metrics_qualities (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            meta_id integer,
                                            acquisition_date text,
                                            quality integer,
                                            resolution_200 integer,
                                            resolution_700 integer,
                                            average_accuracy integer,
                                            chemical_dirt integer,
                                            instrument_noise integer,
                                            isotopic_presence integer,
                                            transmission integer,
                                            fragmentation_305 integer,
                                            fragmentation_712 integer,
                                            baseline_25_150 integer,
                                            baseline_50_150 integer,
                                            baseline_25_650 integer,
                                            baseline_50_650 integer,
                                            sig integer,
                                            s2b integer,
                                            s2n integer,
                                            constraint fk_meta foreign key (meta_id) 
                                                references qc_meta(id) 
                                                on delete cascade 
                                                on update cascade  
                                        ); """

    # create a database connection
    qc_database = create_mysql_connection()

    # NOTE: no idea whether it works for MySQL
    qc_database.execute('pragma journal_mode=wal')

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_metrics_table)
        create_table(qc_database, sql_create_qc_metrics_qualities_table)
    else:
        print("Error! Cannot create database connection.")


def create_qc_tunes_database(new_qc_run):
    """ Adapted from sqlite_connector.py. """

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            md5 text,
                                            processing_date text,
                                            acquisition_date text,
                                            instr integer,
                                            quality integer,
                                            user text,
                                            user_comment text,
                                            chemical_mix_id text,
                                            buffer_id text,
                                            msfe_version text,
                                            norm_scan_1 integer,
                                            norm_scan_2 integer,
                                            norm_scan_3 integer,
                                            chem_scan_1 integer,
                                            inst_scan_1 integer
                                        ); """

    sql_create_qc_tunes_table = """ CREATE TABLE IF NOT EXISTS qc_tunes (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            meta_id integer,
                                            acquisition_date text,
                                            quality integer,
                                            """

    ending = """ constraint fk_meta foreign key (meta_id) 
                        references qc_meta(id) 
                        on delete cascade 
                        on update cascade  
                ); """

    # not readable since there are thousands of features
    sql_create_qc_tunes_table += " text,\n".join(new_qc_run['tunes_names'][0]).replace(":", "_").replace(".", "_")\
                                     .replace("(", "_").replace(")", "").replace("/", "_") + " text,\n "
    sql_create_qc_tunes_table += " real,\n".join(new_qc_run['tunes_names'][1]).replace(":", "_").replace(".", "_")\
                                     .replace("(", "_").replace(")", "").replace("/", "_") + " real,\n "
    sql_create_qc_tunes_table += " real,\n".join(new_qc_run['tunes_names'][2]).replace(":", "_").replace(".", "_")\
                                     .replace("(", "_").replace(")", "").replace("/", "_") + " real,\n " + ending

    # create a database connection
    qc_database = create_mysql_connection()

    # NOTE: no idea whether it works for MySQL
    qc_database.execute('pragma journal_mode=wal')

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_tunes_table)
    else:
        print("Error! Cannot create database connection.")


def create_qc_features_database(new_qc_run):
    """ Adapted from sqlite_connector.py. """

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            md5 text,
                                            processing_date text,
                                            acquisition_date text,
                                            instr integer,
                                            quality integer,
                                            user text,
                                            user_comment text,
                                            chemical_mix_id text,
                                            buffer_id text,
                                            msfe_version text,
                                            norm_scan_1 integer,
                                            norm_scan_2 integer,
                                            norm_scan_3 integer,
                                            chem_scan_1 integer,
                                            inst_scan_1 integer
                                        ); """

    # compose same sql query for qc features (first part) - splitting in parts because SQLite has at max 2000 cols
    sql_create_qc_features_1_table = """ CREATE TABLE IF NOT EXISTS qc_features_1 (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            meta_id integer,
                                            acquisition_date text,
                                            quality integer, 
                                            """

    # compose same sql query for qc features (second part)
    sql_create_qc_features_2_table = """ CREATE TABLE IF NOT EXISTS qc_features_2 (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            meta_id integer,
                                            acquisition_date text,
                                            quality integer, 
                                            """

    ending = """ constraint fk_meta foreign key (meta_id) 
                    references qc_meta(id) 
                    on delete cascade 
                    on update cascade  
                   ); """

    # not readable since there are thousands of features
    sql_create_qc_features_1_table += " real,\n".join(new_qc_run["features_names"][:1996]) + " real,\n " + ending
    sql_create_qc_features_2_table += " real,\n".join(new_qc_run["features_names"][1996:]) + " real,\n " + ending

    # create a database connection
    qc_database = create_mysql_connection()

    # debug: add journal mode that allows multiple users interaction
    qc_database.execute('pragma journal_mode=wal')

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_features_1_table)
        create_table(qc_database, sql_create_qc_features_2_table)
    else:
        print("Error! Cannot create database connection.")


def create_and_fill_qc_metrics_database(new_qc_run, in_debug_mode=False):
    """ Adapted from sqlite_connector.py:
        as of June 15, 2021, we don't push features and tunes to the MySQL db.

        This method creates a new QC database. """

    create_qc_metrics_database()  # values of qc metrics
    qc_metrics_database = create_mysql_connection()

    # inserting values into the new database
    last_row_number_1 = insert_qc_meta(qc_metrics_database, new_qc_run)
    last_row_number_2 = insert_qc_metrics(qc_metrics_database, new_qc_run, last_row_number_1)
    last_row_number_5 = insert_qc_metrics_qualities(qc_metrics_database, new_qc_run, last_row_number_1)

    if in_debug_mode:
        print("inserted: meta:", last_row_number_1, 'metrics:', last_row_number_2, 'qualities:', last_row_number_5)


def set_update_time(db):
    """ This method sets update time for the table to trigger Shiny web-server services. """

    sql = 'UPDATE lcms_tables SET UPDATE_TIME="' \
          + datetime.datetime.now(tz=None).strftime("%Y-%m-%d %H:%M:%S") \
          + '" WHERE TABLE_NAME = "qc_meta"'

    cur = db.cursor()
    cur.execute(sql)
    db.commit()


def insert_new_qc_metrics(qc_run, in_debug_mode=False):
    """ Adapted from sqlite_connector.py.
        This method udpates MySQL database with new QC metrics (quality indicators). """

    qc_metrics_database = create_mysql_connection()

    # inserting qc metrics into a database
    last_row_number_1 = insert_qc_meta(qc_metrics_database, qc_run)
    last_row_number_2 = insert_qc_metrics(qc_metrics_database, qc_run, last_row_number_1)
    last_row_number_5 = insert_qc_metrics_qualities(qc_metrics_database, qc_run, last_row_number_1)

    set_update_time(qc_metrics_database)

    if in_debug_mode:
        print("mysql: inserted 1 row at position: meta:", last_row_number_1, 'metrics:', last_row_number_2, 'qualities:', last_row_number_5)


