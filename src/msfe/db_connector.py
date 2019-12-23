import os, sqlite3, json
from src.msfe.constants import qc_matrix_file_path, qc_database_path


def create_connection(db_file):
    """ Creates a database connection to the SQLite database specified by db_file. """

    db = None
    try:
        db = sqlite3.connect(db_file)
        return db
    except Exception as e:
        print(e)

    return db


def create_table(db, create_table_sql):
    """ Creates a table from the create_table_sql statement. """
    try:
        c = db.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print("Error creating table:", e)


def fetch_table(conn, table_name):
    """ Gets data from the table_name given connection. """
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + table_name)
    colnames = [description[0] for description in cur.description]

    return cur.fetchall(), colnames


def insert_qc_metrics(db, qc_run):
    """ Adds last runs QC  metrics values to the table. """

    qc_metrics = (
        qc_run['acquisition_date'],
        qc_run['quality'],
        *qc_run['metrics_values']
    )

    sql = ''' INSERT INTO qc_metrics(acquisition_date,quality,resolution_200,resolution_700,
                                    average_accuracy,chemical_dirt,instrument_noise,isotopic_presence,
                                    transmission,fragmentation_305,fragmentation_712,baseline_25_150,
                                    baseline_50_150,baseline_25_650,baseline_50_650,signal,
                                    s2b,s2n)
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_metrics)
    db.commit()

    return cur.lastrowid


def insert_qc_features(db, new_qc_run):
    """ Adds last runs QC features values to two tables: features are splitted in two parts,
        since SQLite only has 2000 cols at max for a single table. """

    qc_features_1 = (
        new_qc_run['acquisition_date'],
        new_qc_run['quality'],
        *new_qc_run['features_values'][:1998]
    )

    qc_features_2 = (
        new_qc_run['acquisition_date'],
        new_qc_run['quality'],
        *new_qc_run['features_values'][1998:]
    )

    # first portion
    sql_1 = """ INSERT INTO qc_features_1(acquisition_date,quality,"""
    sql_1 += ",".join(new_qc_run["features_names"][:1998]) + """ ) VALUES("""  # features names
    sql_1 += ",".join(["?" for x in range(2000)]) + """) """  # 2000 question marks

    # second portion
    sql_2 = """ INSERT INTO qc_features_2(acquisition_date,quality,"""
    sql_2 += ",".join(new_qc_run["features_names"][1998:]) + """ ) VALUES("""  # features names
    sql_2 += ",".join(["?" for x in range(len(new_qc_run['features_values']) - 1998 + 2)]) + """) """  # question marks for the rest of values

    cur = db.cursor()
    cur.execute(sql_1, qc_features_1)
    cur.execute(sql_2, qc_features_2)
    db.commit()

    return cur.lastrowid


def insert_qc_meta(db, qc_run):
    """ Adds last runs meta info to the table. """

    qc_meta = (
        qc_run['processing_date'],
        qc_run['acquisition_date'],
        qc_run['quality'],
        qc_run['user_comment'],
        qc_run['chemical_mix_id'],
        qc_run['msfe_version'],
        qc_run['scans_processed']['normal'][0],
        qc_run['scans_processed']['normal'][1],
        qc_run['scans_processed']['normal'][2],
        qc_run['scans_processed']['chemical_noise'][0],
        qc_run['scans_processed']['instrument_noise'][0]
    )

    sql = ''' INSERT INTO qc_meta(processing_date,acquisition_date,quality,user_comment,
                                  chemical_mix_id,msfe_version,norm_scan_1,norm_scan_2,
                                  norm_scan_3,chem_scan_1,inst_scan_1)
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_meta)
    db.commit()

    return cur.lastrowid


def create_qc_database(new_qc_run):

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            processing_date text,
                                            acquisition_date text PRIMARY KEY,
                                            quality integer,
                                            user_comment text,
                                            chemical_mix_id integer,
                                            msfe_version text,
                                            norm_scan_1 integer,
                                            norm_scan_2 integer,
                                            norm_scan_3 integer,
                                            chem_scan_1 integer,
                                            inst_scan_1 integer
                                        ); """

    sql_create_qc_metrics_table = """ CREATE TABLE IF NOT EXISTS qc_metrics (
                                            acquisition_date text PRIMARY KEY,
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
                                            signal integer,
                                            s2b real,
                                            s2n real 
                                        ); """

    # compose same sql query for qc features (first part) - splitting in parts because SQLite has at max 2000 cols
    sql_create_qc_features_1_table = """ CREATE TABLE IF NOT EXISTS qc_features_1 (
                                            acquisition_date text PRIMARY KEY,
                                            quality integer, 
                                            """

    # compose same sql query for qc features (second part)
    sql_create_qc_features_2_table = """ CREATE TABLE IF NOT EXISTS qc_features_2 (
                                            acquisition_date text PRIMARY KEY,
                                            quality integer, 
                                            """

    # not readable since there are thousands of features
    sql_create_qc_features_1_table += " real,\n".join(new_qc_run["features_names"][:1998]) + " real\n ); """
    sql_create_qc_features_2_table += " real,\n".join(new_qc_run["features_names"][1998:]) + " real\n ); """

    # create a database connection
    qc_database = create_connection(qc_database_path)

    # debug: add journal mode that allows multiple users interaction
    qc_database.execute('pragma journal_mode=wal')

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_metrics_table)
        create_table(qc_database, sql_create_qc_features_1_table)
        create_table(qc_database, sql_create_qc_features_2_table)
    else:
        print("Error! Cannot create database connection.")


def create_and_fill_qc_database(new_qc_run, in_debug_mode=False):
    """ This method creates a new QC database out of a qc_matrix object. """

    create_qc_database(new_qc_run)
    qc_database = create_connection(qc_database_path)

    # inserting values into the new database
    last_row_number_1 = insert_qc_meta(qc_database, new_qc_run)
    last_row_number_2 = insert_qc_metrics(qc_database, new_qc_run)
    last_row_number_3 = insert_qc_features(qc_database, new_qc_run)

    if in_debug_mode:
        print("inserted: meta:", last_row_number_1, 'metrics:', last_row_number_2, 'features:', last_row_number_3)


def insert_new_qc_run(qc_run, in_debug_mode=False):
    """ This method form objects with pre-computed values to insert into (already existing) database. """

    qc_database = create_connection(qc_database_path)

    # inserting values into the new database
    last_row_number_1 = insert_qc_meta(qc_database, qc_run)
    last_row_number_2 = insert_qc_metrics(qc_database, qc_run)
    last_row_number_3 = insert_qc_features(qc_database, qc_run)

    if in_debug_mode:
        print("inserted 1 row at position: meta:", last_row_number_1, 'metrics:', last_row_number_2, "features:", last_row_number_3)


if __name__ == '__main__':

    pass

