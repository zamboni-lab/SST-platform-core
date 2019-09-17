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
        print(e)


def insert_qc_values(db, qc_values):
    """ Adds last runs QC values to the table. """

    sql = ''' INSERT INTO qc_values(acquisition_date,quality,resolution_200,resolution_700,
                                    average_accuracy,chemical_dirt,instrument_noise,isotopic_presence,
                                    transmission,fragmentation_305,fragmentation_712,baseline_25_150,
                                    baseline_50_150,baseline_25_650,baseline_50_650,signal,
                                    s2b,s2n)
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_values)
    db.commit()

    return cur.lastrowid


def insert_qc_meta(db, qc_meta):
    """ Adds last runs meta info to the table. """

    sql = ''' INSERT INTO qc_meta(processing_date,acquisition_date,quality,user_comment,
                                  chemical_mix_id,msfe_version,norm_scan_1,norm_scan_2,
                                  norm_scan_3,chem_scan_1,inst_scan_1)
                          
                          VALUES(?,?,?,?,?,?,?,?,?,?,?) '''

    cur = db.cursor()
    cur.execute(sql, qc_meta)
    db.commit()

    return cur.lastrowid


def create_qc_database(db_path='/Users/andreidm/ETH/projects/qc_metrics/res/qc_matrix.db'):

    sql_create_qc_meta_table = """ CREATE TABLE IF NOT EXISTS qc_meta (
                                            processing_date text PRIMARY KEY,
                                            acquisition_date text,
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

    sql_create_qc_values_table = """ CREATE TABLE IF NOT EXISTS qc_values (
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

    # create a database connection
    qc_database = create_connection(qc_database_path)

    # create tables
    if qc_database is not None:
        # create projects table
        create_table(qc_database, sql_create_qc_meta_table)
        create_table(qc_database, sql_create_qc_values_table)
    else:
        print("Error! cannot create the database connection.")


def create_and_fill_qc_database(qc_matrix, debug=False):
    """ This method creates a new QC database out of a qc_matrix object. """

    create_qc_database(qc_database_path)
    qc_database = create_connection(qc_database_path)

    for qc_run in qc_matrix['qc_runs']:

        run_meta = (
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

        run_values = (
            qc_run['acquisition_date'],
            qc_run['quality'],
            *qc_run['qc_values']
        )

        # inserting values into the new database
        last_row_number_1 = insert_qc_meta(qc_database, run_meta)
        last_row_number_2 = insert_qc_values(qc_database, run_values)

        if debug:
            print("inserted: meta:", last_row_number_1, 'values:', last_row_number_2)


def insert_new_qc_run(qc_run, in_debug_mode=False):
    """ This method form objects with pre-computed values to insert into (already existing) database. """

    qc_database = create_connection(qc_database_path)

    run_meta = (
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

    run_values = (
        qc_run['acquisition_date'],
        qc_run['quality'],
        *qc_run['qc_values']
    )

    # inserting values into the new database
    last_row_number_1 = insert_qc_meta(qc_database, run_meta)
    last_row_number_2 = insert_qc_values(qc_database, run_values)

    if in_debug_mode:
        print("inserted 1 row at position: meta:", last_row_number_1, 'values:', last_row_number_2)


if __name__ == '__main__':

    pass

