import sqlite3
from src.constants import qc_metrics_database_path, qc_features_database_path, qc_tunes_database_path
from src.qcmg import metrics_generator


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


def insert_qc_metrics(db, qc_run, meta_id):
    """ Adds last runs QC  metrics values to the table. """

    qc_metrics = (
        meta_id,
        qc_run['acquisition_date'],
        qc_run['quality'],
        *qc_run['metrics_values']
    )

    sql = ''' INSERT INTO qc_metrics(meta_id,acquisition_date,quality,resolution_200,resolution_700,
                                    average_accuracy,chemical_dirt,instrument_noise,isotopic_presence,
                                    transmission,fragmentation_305,fragmentation_712,baseline_25_150,
                                    baseline_50_150,baseline_25_650,baseline_50_650,signal,
                                    s2b,s2n)
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_metrics)
    db.commit()

    return cur.lastrowid


def insert_qc_metrics_qualities(db, qc_run, meta_id):
    """ Adds last run QC metrics qualities (0 or 1 per each metric) to the table. """

    qc_metrics = (
        meta_id,
        qc_run['acquisition_date'],
        qc_run['quality'],
        *qc_run['metrics_qualities']
    )

    sql = ''' INSERT INTO qc_metrics_qualities(meta_id,acquisition_date,quality,resolution_200,resolution_700,
                                    average_accuracy,chemical_dirt,instrument_noise,isotopic_presence,
                                    transmission,fragmentation_305,fragmentation_712,baseline_25_150,
                                    baseline_50_150,baseline_25_650,baseline_50_650,signal,
                                    s2b,s2n)
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_metrics)
    db.commit()

    return cur.lastrowid


def insert_qc_features(db, new_qc_run, meta_id):
    """ Adds last runs QC features values to two tables: features are splitted in two parts,
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
    sql_1 += ",".join(["?" for x in range(1999)]) + """) """  # 1999 question marks

    # second portion
    sql_2 = """ INSERT INTO qc_features_2(meta_id,acquisition_date,quality,"""
    sql_2 += ",".join(new_qc_run["features_names"][1996:]) + """ ) VALUES("""  # features names
    sql_2 += ",".join(["?" for x in range(len(new_qc_run['features_values']) - 1996 + 3)]) + """) """  # question marks for the rest of values

    cur = db.cursor()
    cur.execute(sql_1, qc_features_1)
    cur.execute(sql_2, qc_features_2)
    db.commit()

    return cur.lastrowid


def insert_qc_tunes(db, new_qc_run, meta_id):
    """ Adds last runs QC tunes to a table. """

    single_names_string = ",".join(new_qc_run['tunes_names'][0]) + ","
    single_names_string += ",".join(new_qc_run['tunes_names'][1]) + ","
    single_names_string += ",".join(new_qc_run['tunes_names'][2])
    single_names_string = single_names_string.replace(":","_").replace(".","_").replace("(","_").replace(")","").replace("/","_")

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
    sql += ",".join(["?" for x in range(len(single_values_list) + 3)]) + """) """  # question marks

    cur = db.cursor()
    cur.execute(sql, qc_tunes)
    db.commit()

    return cur.lastrowid


def insert_qc_meta(db, qc_run):
    """ Adds last runs meta info to the table. """

    qc_meta = (
        qc_run['md5'],
        qc_run['processing_date'],
        qc_run['acquisition_date'],
        qc_run['instrument'],
        qc_run['quality'],
        qc_run['user'],
        qc_run['user_comment'],
        qc_run['chemical_mix_id'],
        qc_run['msfe_version'],
        qc_run['scans_processed']['normal'][0],
        qc_run['scans_processed']['normal'][1],
        qc_run['scans_processed']['normal'][2],
        qc_run['scans_processed']['chemical_noise'][0],
        qc_run['scans_processed']['instrument_noise'][0]
    )

    sql = ''' INSERT INTO qc_meta(md5,processing_date,acquisition_date,instr,quality,user,user_comment,
                                  chemical_mix_id,msfe_version,norm_scan_1,norm_scan_2,
                                  norm_scan_3,chem_scan_1,inst_scan_1)
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_meta)
    db.commit()

    return cur.lastrowid


def create_qc_metrics_database():

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            md5 text,
                                            processing_date text,
                                            acquisition_date text,
                                            instr integer,
                                            quality integer,
                                            user text,
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
                                            signal integer,
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
                                            signal integer,
                                            s2b integer,
                                            s2n integer,
                                            constraint fk_meta foreign key (meta_id) 
                                                references qc_meta(id) 
                                                on delete cascade 
                                                on update cascade  
                                        ); """

    # create a database connection
    qc_database = create_connection(qc_metrics_database_path)

    # debug: add journal mode that allows multiple users interaction
    qc_database.execute('pragma journal_mode=wal')

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_metrics_table)
        create_table(qc_database, sql_create_qc_metrics_qualities_table)
    else:
        print("Error! Cannot create database connection.")


def create_qc_metrics_qualities_table(db_path):
    """ This method adds just one table - QC metrics qualities - to the existing database. """

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
                                                signal integer,
                                                s2b integer,
                                                s2n integer,
                                                constraint fk_meta foreign key (meta_id) 
                                                    references qc_meta(id) 
                                                    on delete cascade 
                                                    on update cascade  
                                            ); """

    # create a database connection
    qc_database = create_connection(db_path)

    # add journal mode that allows multiple users interaction
    qc_database.execute('pragma journal_mode=wal')

    if qc_database is not None:
        # create table
        create_table(qc_database, sql_create_qc_metrics_qualities_table)
    else:
        print("Error! Cannot create database connection.")


def create_qc_tunes_database(new_qc_run):

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            md5 text,
                                            processing_date text,
                                            acquisition_date text,
                                            instr integer,
                                            quality integer,
                                            user text,
                                            user_comment text,
                                            chemical_mix_id integer,
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
    sql_create_qc_tunes_table += " text,\n".join(new_qc_run['tunes_names'][0]).replace(":","_").replace(".","_").replace("(","_").replace(")","").replace("/","_") + " text,\n "
    sql_create_qc_tunes_table += " real,\n".join(new_qc_run['tunes_names'][1]).replace(":","_").replace(".","_").replace("(","_").replace(")","").replace("/","_") + " real,\n "
    sql_create_qc_tunes_table += " real,\n".join(new_qc_run['tunes_names'][2]).replace(":","_").replace(".","_").replace("(","_").replace(")","").replace("/","_") + " real,\n " + ending

    # create a database connection
    qc_database = create_connection(qc_tunes_database_path)

    # debug: add journal mode that allows multiple users interaction
    qc_database.execute('pragma journal_mode=wal')

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_tunes_table)
    else:
        print("Error! Cannot create database connection.")


def create_qc_features_database(new_qc_run):

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            md5 text,
                                            processing_date text,
                                            acquisition_date text,
                                            instr integer,
                                            quality integer,
                                            user text,
                                            user_comment text,
                                            chemical_mix_id integer,
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
    qc_database = create_connection(qc_features_database_path)

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


def create_and_fill_qc_databases(new_qc_run, in_debug_mode=False):
    """ This method creates a new QC database out of a qc_matrix object. """

    create_qc_metrics_database()  # values of qc metrics
    qc_metrics_database = create_connection(qc_metrics_database_path)

    # inserting values into the new database
    last_row_number_1 = insert_qc_meta(qc_metrics_database, new_qc_run)
    last_row_number_2 = insert_qc_metrics(qc_metrics_database, new_qc_run, last_row_number_1)

    create_qc_features_database(new_qc_run)  # values of qc features
    qc_features_database = create_connection(qc_features_database_path)

    last_row_number_1 = insert_qc_meta(qc_features_database, new_qc_run)
    last_row_number_3 = insert_qc_features(qc_features_database, new_qc_run, last_row_number_1)

    create_qc_tunes_database(new_qc_run)  # values of instrument settings
    qc_tunes_database = create_connection(qc_tunes_database_path)

    last_row_number_1 = insert_qc_meta(qc_tunes_database, new_qc_run)
    last_row_number_4 = insert_qc_tunes(qc_tunes_database, new_qc_run, last_row_number_1)

    if in_debug_mode:
        print("inserted: meta:", last_row_number_1, 'metrics:', last_row_number_2, 'features:', last_row_number_3, 'tunes:', last_row_number_4)


def insert_new_qc_run(qc_run, in_debug_mode=False):
    """ This method form objects with pre-computed values to insert into (already existing) database. """

    qc_metrics_database = create_connection(qc_metrics_database_path)

    # inserting qc metrics into a database
    last_row_number_1 = insert_qc_meta(qc_metrics_database, qc_run)
    last_row_number_2 = insert_qc_metrics(qc_metrics_database, qc_run, last_row_number_1)

    qc_features_database = create_connection(qc_features_database_path)

    # inserting qc features into another database
    last_row_number_1 = insert_qc_meta(qc_features_database, qc_run)
    last_row_number_3 = insert_qc_features(qc_features_database, qc_run, last_row_number_1)

    qc_tunes_database = create_connection(qc_tunes_database_path)

    # inserting qc features into a third database
    last_row_number_1 = insert_qc_meta(qc_tunes_database, qc_run)
    last_row_number_4 = insert_qc_tunes(qc_tunes_database, qc_run, last_row_number_1)

    if in_debug_mode:
        print("inserted 1 row at position: meta:", last_row_number_1, 'metrics:', last_row_number_2, "features:", last_row_number_3, 'tunes:', last_row_number_4)


def add_qc_metrics_qualities_table(db_path):
    """ This method creates and adds a new table to existing database,
        row by row, calling inserting method iteratively. """

    # get quality table for existing database
    quality_table = metrics_generator.compute_quality_table(db_path)

    create_qc_metrics_qualities_table(db_path)  # create table in existing database
    qc_metrics_database = create_connection(db_path)

    for i in range(quality_table.shape[0]):

        # make artificial packing to use the same method
        meta_id = int(quality_table.iloc[i, 1])
        qc_run = {
            "acquisition_date": str(quality_table.iloc[i, 2]),
            "quality": int(quality_table.iloc[i, 3]),
            "metrics_qualities": [int(x) for x in quality_table.iloc[i, 4:]]
        }

        last_row_number = insert_qc_metrics_qualities(qc_metrics_database, qc_run, meta_id)

        print("inserted:", last_row_number)


if __name__ == '__main__':

    db_path = "/Users/andreidm/ETH/projects/shiny_qc/data/nas2_qc_metrics_database_mar18.sqlite"
    add_qc_metrics_qualities_table(db_path)





    pass

