import sqlite3, pandas
from src.constants import qc_metrics_database_path, qc_features_database_path, qc_tunes_database_path, all_metrics
from src.qcmg import metrics_generator
from src.constants import get_buffer_id


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


def update_column_in_database(conn, table_name, colname, col_values, rowname, row_values):
    """ This method updates the whole quality column in qc_meta table of the given database.
        Called to update QC features and QC tunes databases,
        when qualities are newly generated for QC metrics database. """

    sql = ''' UPDATE table_name
              SET column_name = ?
              WHERE row_name = ? '''

    sql = sql.replace("table_name", table_name).replace("column_name", colname).replace("row_name", rowname)

    cur = conn.cursor()

    multiple_entries = [(col_values[i], row_values[i]) for i in range(len(row_values))]

    cur.executemany(sql, multiple_entries)
    conn.commit()


def add_column_to_database(conn, table_name, column_name, column_type):
    """ This method adds a new column of given name and type to an existing database. """

    sql = ''' ALTER TABLE table_name
              ADD COLUMN column_name column_type '''

    sql = sql.replace("table_name", table_name).replace("column_name", column_name).replace("column_type", column_type)

    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()


def remove_row_from_table_by_id(conn, table_name, id):
    """ This method removes a row of given id from an existing database. """

    sql = ''' DELETE FROM table_name
              WHERE id=? '''

    sql = sql.replace("table_name", table_name).replace("?", str(id))

    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()


def remove_row_from_all_databases_by_id(run_id,
                                        features_db_path=qc_features_database_path,
                                        metrics_db_path=qc_metrics_database_path,
                                        tunes_db_path=qc_tunes_database_path):
    """ This method removes a row with a given id from all tables of all databases. """

    conn = create_connection(features_db_path)
    remove_row_from_table_by_id(conn, 'qc_features_1', run_id)
    remove_row_from_table_by_id(conn, 'qc_features_2', run_id)
    remove_row_from_table_by_id(conn, 'qc_meta', run_id)

    conn = create_connection(metrics_db_path)
    remove_row_from_table_by_id(conn, 'qc_metrics', run_id)
    remove_row_from_table_by_id(conn, 'qc_metrics_qualities', run_id)
    remove_row_from_table_by_id(conn, 'qc_meta', run_id)

    conn = create_connection(tunes_db_path)
    remove_row_from_table_by_id(conn, 'qc_tunes', run_id)
    remove_row_from_table_by_id(conn, 'qc_meta', run_id)


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
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_meta)
    db.commit()

    return cur.lastrowid


def create_qc_metrics_database(metrics_db_path):

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
    qc_database = create_connection(metrics_db_path)

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


def create_qc_metrics_qualities_table(qc_database):
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

    create_table(qc_database, sql_create_qc_metrics_qualities_table)


def create_qc_tunes_database(new_qc_run, tunes_db_path):

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
    sql_create_qc_tunes_table += " text,\n".join(new_qc_run['tunes_names'][0]).replace(":","_").replace(".","_").replace("(","_").replace(")","").replace("/","_") + " text,\n "
    sql_create_qc_tunes_table += " real,\n".join(new_qc_run['tunes_names'][1]).replace(":","_").replace(".","_").replace("(","_").replace(")","").replace("/","_") + " real,\n "
    sql_create_qc_tunes_table += " real,\n".join(new_qc_run['tunes_names'][2]).replace(":","_").replace(".","_").replace("(","_").replace(")","").replace("/","_") + " real,\n " + ending

    # create a database connection
    qc_database = create_connection(tunes_db_path)

    # debug: add journal mode that allows multiple users interaction
    qc_database.execute('pragma journal_mode=wal')

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_tunes_table)
    else:
        print("Error! Cannot create database connection.")


def create_qc_features_database(new_qc_run, features_db_path):

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
    qc_database = create_connection(features_db_path)

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


def create_and_fill_qc_databases(new_qc_run,
                                 metrics_db_path=qc_metrics_database_path,
                                 features_db_path=qc_features_database_path,
                                 tunes_db_path=qc_tunes_database_path,
                                 in_debug_mode=False):
    """ This method creates a new QC database out of a qc_matrix object. """

    create_qc_metrics_database(metrics_db_path)  # values of qc metrics
    qc_metrics_database = create_connection(metrics_db_path)

    # inserting values into the new database
    last_row_number_1 = insert_qc_meta(qc_metrics_database, new_qc_run)
    last_row_number_2 = insert_qc_metrics(qc_metrics_database, new_qc_run, last_row_number_1)
    last_row_number_5 = insert_qc_metrics_qualities(qc_metrics_database, new_qc_run, last_row_number_1)

    create_qc_features_database(new_qc_run, features_db_path)  # values of qc features
    qc_features_database = create_connection(features_db_path)

    last_row_number_1 = insert_qc_meta(qc_features_database, new_qc_run)
    last_row_number_3 = insert_qc_features(qc_features_database, new_qc_run, last_row_number_1)

    create_qc_tunes_database(new_qc_run, tunes_db_path)  # values of instrument settings
    qc_tunes_database = create_connection(tunes_db_path)

    last_row_number_1 = insert_qc_meta(qc_tunes_database, new_qc_run)
    last_row_number_4 = insert_qc_tunes(qc_tunes_database, new_qc_run, last_row_number_1)

    if in_debug_mode:
        print("inserted: meta:", last_row_number_1, 'metrics:', last_row_number_2, 'qualities:', last_row_number_5, 'features:', last_row_number_3, 'tunes:', last_row_number_4)


def insert_new_qc_run(qc_run,
                      metrics_db_path=qc_metrics_database_path,
                      features_db_path=qc_features_database_path,
                      tunes_db_path=qc_tunes_database_path,
                      in_debug_mode=False):
    """ This method form objects with pre-computed values to insert into (already existing) database. """

    qc_metrics_database = create_connection(metrics_db_path)

    # inserting qc metrics into a database
    last_row_number_1 = insert_qc_meta(qc_metrics_database, qc_run)
    last_row_number_2 = insert_qc_metrics(qc_metrics_database, qc_run, last_row_number_1)
    last_row_number_5 = insert_qc_metrics_qualities(qc_metrics_database, qc_run, last_row_number_1)

    qc_features_database = create_connection(features_db_path)

    # inserting qc features into another database
    last_row_number_1 = insert_qc_meta(qc_features_database, qc_run)
    last_row_number_3 = insert_qc_features(qc_features_database, qc_run, last_row_number_1)

    qc_tunes_database = create_connection(tunes_db_path)

    # inserting qc features into a third database
    last_row_number_1 = insert_qc_meta(qc_tunes_database, qc_run)
    last_row_number_4 = insert_qc_tunes(qc_tunes_database, qc_run, last_row_number_1)

    if in_debug_mode:
        print("inserted 1 row at position: meta:", last_row_number_1, 'metrics:', last_row_number_2, 'qualities:', last_row_number_5, "features:", last_row_number_3, 'tunes:', last_row_number_4)


def add_quality_column_to_databases(qualities, meta_ids):
    """ This methods updates the quality column (summed over all metrics) to all the databases. """

    # connect to all dbs
    metrics_db = create_connection(qc_metrics_database_path)
    features_db = create_connection(qc_features_database_path)
    tunes_db = create_connection(qc_tunes_database_path)

    # update all tables with qualities
    update_column_in_database(metrics_db, "qc_meta", "quality", qualities, "id", meta_ids)
    update_column_in_database(metrics_db, "qc_metrics", "quality", qualities, "meta_id", meta_ids)
    update_column_in_database(features_db, "qc_meta", "quality", qualities, "id", meta_ids)
    update_column_in_database(features_db, "qc_features_1", "quality", qualities, "meta_id", meta_ids)
    update_column_in_database(features_db, "qc_features_2", "quality", qualities, "meta_id", meta_ids)
    update_column_in_database(tunes_db, "qc_meta", "quality", qualities, "id", meta_ids)
    update_column_in_database(tunes_db, "qc_tunes", "quality", qualities, "meta_id", meta_ids)


def update_all_databases_with_qualities(quality_table, metrics_data,
                                        metrics_db_path=qc_metrics_database_path,
                                        features_db_path=qc_features_database_path,
                                        tunes_db_path=qc_tunes_database_path):
    """ This methods updates the quality column (summed over all metrics) and qualities for each QC metric
        to all the databases. """

    meta_ids = [int(x) for x in metrics_data['meta_id'].tolist()]
    main_qualities = [int(x) for x in quality_table['quality'].tolist()]

    # connect to all dbs
    metrics_db = create_connection(metrics_db_path)
    features_db = create_connection(features_db_path)
    tunes_db = create_connection(tunes_db_path)

    # update qc_metrics_qualities table in qc_metrics database
    for metric_name in all_metrics:
        # make a list of qualities for this metric (excluding the last run for now)
        metric_qualities = [int(x) for x in quality_table[metric_name].tolist()]
        update_column_in_database(metrics_db, "qc_metrics_qualities", metric_name, metric_qualities, "meta_id", meta_ids)

    # update all tables with qualities
    update_column_in_database(metrics_db, "qc_meta", "quality", main_qualities, "id", meta_ids)
    update_column_in_database(metrics_db, "qc_metrics", "quality", main_qualities, "meta_id", meta_ids)
    update_column_in_database(features_db, "qc_meta", "quality", main_qualities, "id", meta_ids)
    update_column_in_database(features_db, "qc_features_1", "quality", main_qualities, "meta_id", meta_ids)
    update_column_in_database(features_db, "qc_features_2", "quality", main_qualities, "meta_id", meta_ids)
    update_column_in_database(tunes_db, "qc_meta", "quality", main_qualities, "id", meta_ids)
    update_column_in_database(tunes_db, "qc_tunes", "quality", main_qualities, "meta_id", meta_ids)


def update_old_databases_with_qualities_and_buffer_info(paths):
    """ This method updates old databases with two new instances: buffer_id and metric-wise quality values.
        First, it adds buffer information based on the date of acquisition (we hardcode in constants dates of changes),
        then, it splits entries of databases by buffer, to compute quality tables for metrics.
        After qualities are computed, it updates all the databases with common quality
                                                                    and adds quality table to QC metrics database."""

    features_db_path = paths[0]
    metrics_db_path = paths[1]
    tunes_db_path = paths[2]

    # fetch QC metrics db
    metrics_db = create_connection(metrics_db_path)
    database, colnames = fetch_table(metrics_db, "qc_metrics")

    data = pandas.DataFrame(database)
    data.columns = colnames

    # add buffer id information to metrics data
    acquisition_dates = data.iloc[:,2]
    buffer_ids = [get_buffer_id(date) for date in acquisition_dates]
    data['buffer_id'] = buffer_ids

    # update all databases
    features_db = create_connection(features_db_path)
    tunes_db = create_connection(tunes_db_path)

    # add only to qc_meta tables
    add_column_to_database(metrics_db, "qc_meta", "buffer_id", "text")
    add_column_to_database(features_db, "qc_meta", "buffer_id", "text")
    add_column_to_database(tunes_db, "qc_meta", "buffer_id", "text")

    update_column_in_database(metrics_db, "qc_meta", "buffer_id", buffer_ids, "acquisition_date", acquisition_dates)
    update_column_in_database(features_db, "qc_meta", "buffer_id", buffer_ids, "acquisition_date", acquisition_dates)
    update_column_in_database(tunes_db, "qc_meta", "buffer_id", buffer_ids, "acquisition_date", acquisition_dates)

    # create new table in existing metrics database
    create_qc_metrics_qualities_table(metrics_db)

    # compute and add qualities for runs, split by different buffers
    for buffer in list(set(buffer_ids)):

        # select entries corresponding to this buffer, remove buffer_id column itself
        buffer_data = data[data['buffer_id'] == buffer].drop(labels='buffer_id', axis=1)

        # some stupid manipulations to reuse the same method
        last_run_metrics = list(buffer_data.iloc[-1, 4:])
        metrics_data = buffer_data.iloc[0:-1, :]

        # get quality table for existing database (calculate for the first time here)
        quality_table = metrics_generator.recompute_quality_table_for_all_runs(last_run_metrics, metrics_data)

        assert buffer_data.shape[0] == quality_table.shape[0]

        meta_ids = [int(x) for x in buffer_data['meta_id'].tolist()]
        acquisition_dates = buffer_data['acquisition_date'].tolist()
        main_qualities = [int(x) for x in quality_table['quality'].tolist()]

        # update databases
        add_quality_column_to_databases(main_qualities, meta_ids)

        for i in range(quality_table.shape[0]):

            # some other manipulations to reuse another method
            qc_run = {
                "acquisition_date": acquisition_dates[i],
                "quality": main_qualities[i],
                "metrics_qualities": [int(x) for x in quality_table.iloc[i, 1:]]
            }

            meta_id = meta_ids[i]

            last_row_number = insert_qc_metrics_qualities(metrics_db, qc_run, meta_id)


if __name__ == '__main__':

    # db_path = "/Users/andreidm/ETH/projects/shiny_qc/data/nas2_qc_metrics_database_apr15.sqlite"
    #
    # # create and fill metric quality table for existing qc_metrics_database
    # conn = create_connection(db_path)
    #
    # qc_metrics, colnames = fetch_table(conn, "qc_metrics")
    # metrics_data = pandas.DataFrame(qc_metrics)
    # metrics_data.columns = colnames
    #
    # qc_meta, colnames = fetch_table(conn, "qc_meta")
    # meta_data = pandas.DataFrame(qc_meta)
    # meta_data.columns = colnames

    paths = [
        "/Users/andreidm/ETH/projects/monitoring_system/res/qc_features_database.sqlite",
        "/Users/andreidm/ETH/projects/monitoring_system/res/qc_metrics_database.sqlite",
        "/Users/andreidm/ETH/projects/monitoring_system/res/qc_tunes_database.sqlite"
    ]

    update_old_databases_with_qualities_and_buffer_info(paths)


    pass

