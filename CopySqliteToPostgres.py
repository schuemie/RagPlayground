import os
import sqlite3

import psycopg
from psycopg import sql
from dotenv import load_dotenv

load_dotenv()


# Configuration
sqlite_db = 'e:/PubMed/PubMed.sqlite'  # Path to SQLite database
chunk_size = 100000  # Number of rows per chunk for large tables

# Connect to SQLite
sqlite_conn = sqlite3.connect(sqlite_db)
sqlite_cursor = sqlite_conn.cursor()

# Connect to PostgreSQL
postgres_conn = psycopg.connect(host=os.getenv("POSTGRES_SERVER"),
                                user=os.getenv("POSTGRES_USER"),
                                password=os.getenv("POSTGRES_PASSWORD"),
                                dbname=os.getenv("POSTGRES_DATABASE"))
postgres_cursor = postgres_conn.cursor()
postgres_cursor.execute(f"SET SEARCH_PATH={os.getenv('POSTGRES_SCHEMA')};")

def create_table_in_postgres(table_name, columns):
    """
    Creates a table in PostgreSQL using the schema from SQLite.
    """
    postgres_cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    column_defs = ', '.join(f"{col['name']} {col['type']}" for col in columns)
    create_table_query = f"CREATE TABLE {table_name} ({column_defs})"
    postgres_cursor.execute(create_table_query)
    postgres_conn.commit()
    print(f"Created table {table_name} in PostgreSQL.")


def transfer_table_data(table_name, columns):
    """
    Transfers data from an SQLite table to PostgreSQL in chunks using fetchmany().
    """
    column_names = ', '.join([col['name'] for col in columns])
    placeholders = ', '.join(['%s'] * len(columns))

    # Prepare the PostgreSQL INSERT query
    insert_query = sql.SQL(f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})")

    # Read and transfer data in chunks
    sqlite_cursor.execute(f"SELECT * FROM {table_name}")
    total_rows = 0
    while True:
        rows = sqlite_cursor.fetchmany(chunk_size)  # Fetch the next chunk of rows

        if not rows:  # Break the loop when no more rows are available
            break

        # Insert rows into PostgreSQL
        postgres_cursor.executemany(insert_query.as_string(postgres_conn), rows)
        postgres_conn.commit()
        total_rows = total_rows + len(rows)
        print(f"Copied {total_rows} rows to table {table_name}.")


def get_column_types_from_sqlite(table_name):
    """
    Fetches column names and types from the SQLite table and converts them to PostgreSQL-compatible types.
    """
    sqlite_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = sqlite_cursor.fetchall()

    column_defs = []
    for column in columns:
        column_name = column[1]
        sqlite_type = column[2].upper()

        # Map SQLite types to PostgreSQL types
        if 'INT' in sqlite_type:
            postgres_type = 'INTEGER'
        elif 'CHAR' in sqlite_type or 'TEXT' in sqlite_type:
            postgres_type = 'TEXT'
        elif 'BLOB' in sqlite_type:
            postgres_type = 'BYTEA'
        elif 'REAL' in sqlite_type or 'FLOA' in sqlite_type or 'DOUB' in sqlite_type:
            postgres_type = 'DOUBLE PRECISION'
        else:
            postgres_type = 'TEXT'  # Default to TEXT for unrecognized types

        column_defs.append({'name': column_name, 'type': postgres_type})

    return column_defs


def transfer_all_tables():
    """
    Copies all tables from the SQLite database to the PostgreSQL database.
    """
    sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = sqlite_cursor.fetchall()

    for (table_name,) in tables:
        print(f"Transferring table: {table_name}")

        # Get columns and create table in PostgreSQL
        columns = get_column_types_from_sqlite(table_name)
        create_table_in_postgres(table_name, columns)

        # Transfer data in chunks
        transfer_table_data(table_name, columns)


# Run the transfer
transfer_all_tables()

# Close connections
sqlite_conn.close()
postgres_conn.close()

# CREATE UNIQUE INDEX idx_pmid_pubmed_articles ON pubmed.pubmed_articles (pmid);
# CREATE INDEX idx_pd_pubmed_articles ON pubmed.pubmed_articles (publication_date);