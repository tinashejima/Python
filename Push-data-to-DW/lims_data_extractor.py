import os
import time
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

# File targets for your Power BI dashboards
FILE_STATISTICS = "marts_lab_lims_statistics.xlsx"
FILE_VIRAL_LOAD = "marts_lims_viral_load_results.xlsx"
TRACKING_FILE = "last_sync_timestamp.txt"

def get_sqlalchemy_engine():
    """Establishes an SQLAlchemy connection engine using variables from .env."""
    try:
        dbname = os.getenv('LIMS_DATAMART_DB_NAME')
        user = os.getenv('LIMS_DATAMART_DB_USERNAME')
        password = os.getenv('LIMS_DATAMART_DB_PASSWORD')
        host = os.getenv('LIMS_DATAMART_HOST')
        port = os.getenv('LIMS_DATAMART_PORT')
        
        if not all([dbname, user, password, host, port]):
            print("[-] Error: Missing one or more LIMS_DATAMART configuration keys in .env file.")
            return None

        connection_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(connection_uri)
        return engine
    except Exception as e:
        print(f"[-] Database engine creation failure: {e}")
        return None

def get_last_synced_timestamp():
    """Gets the tracking state timestamp."""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return f.read().strip()
    return "2025-02-18 00:00:00"

def save_last_synced_timestamp(timestamp):
    """Saves the last processed timestamp state."""
    with open(TRACKING_FILE, "w") as f:
        f.write(str(timestamp))

def update_excel_file(file_path, new_df, key_column):
    """Upserts incoming data modifications into the targeted Power BI Excel sheet and logs row actions."""
    if new_df.empty:
        return

    # Work on a copy and set the index to track matchings
    new_df_indexed = new_df.copy().set_index(key_column)

    if os.path.exists(file_path):
        try:
            print(f"[+] Syncing live modifications into: {file_path}")
            existing_df = pd.read_excel(file_path, engine='openpyxl')
            existing_df.set_index(key_column, inplace=True)

            # --- Trace and Log Actions Row-by-Row ---
            for record_id in new_df_indexed.index:
                if record_id in existing_df.index:
                    print(f"Updating record for {key_column}: {record_id}")
                else:
                    print(f"Inserting new record for {key_column}: {record_id}")

            # 1. Overwrite rows that have status changes or new results
            existing_df.update(new_df_indexed)

            # 2. Append completely fresh records that do not exist yet
            brand_new_records = new_df_indexed[~new_df_indexed.index.isin(existing_df.index)]
            final_df = pd.concat([existing_df, brand_new_records])
        except Exception as e:
            print(f"[!] Error reading file, fallback to recreating it: {e}")
            final_df = new_df_indexed
    else:
        print(f"[!] Target file {file_path} not found. Creating a fresh master file...")
        # Every entry is an insertion if the master file does not exist yet
        for record_id in new_df_indexed.index:
            print(f"Inserting new record for {key_column}: {record_id}")
        final_df = new_df_indexed

    # Reset index back to regular layout and output for Power BI consumption
    final_df.reset_index(inplace=True)
    final_df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"[✓] File sync complete. Total active records: {len(final_df)}")

def check_and_sync_changes(engine, last_timestamp):
    """Queries Postgres using SQLAlchemy engine for changes and updates Excel files."""
    
    with engine.connect() as connection:
        # --- 1. Processing Table: lab_lims_statistics ---
        query_stats = """
            SELECT * FROM marts.lab_lims_statistics 
            WHERE dm_date_created > :last_timestamp 
            ORDER BY dm_date_created ASC
        """
        df_stats = pd.read_sql_query(text(query_stats), connection, params={"last_timestamp": last_timestamp})

        # --- 2. Processing View: lims_viral_load_results ---
        query_vl = """
            SELECT * FROM marts.lims_viral_load_results 
            WHERE shr_date >= CAST(:last_date AS DATE)
        """
        df_vl = pd.read_sql_query(text(query_vl), connection, params={"last_date": last_timestamp[:10]})

    # Track max dynamic timestamp to move forward our delta synchronization bookmark
    max_timestamp = last_timestamp

    # Sync File 1: Statistics
    if not df_stats.empty:
        update_excel_file(FILE_STATISTICS, df_stats, key_column='encounter_id')
        if 'dm_date_created' in df_stats.columns:
            latest_db_record = df_stats['dm_date_created'].max()
            max_timestamp = str(latest_db_record)

    # Sync File 2: Viral Load Results (With smart ID detection)
    if not df_vl.empty:
        if 'encounter_id' in df_vl.columns:
            vl_key = 'encounter_id'
        elif 'encounter_lab_request' in df_vl.columns:
            vl_key = 'encounter_lab_request'
        elif 'lab_request_number' in df_vl.columns:
            vl_key = 'lab_request_number'
        else:
            vl_key = df_vl.columns[0]
            print(f"[!] Warning: Unknown ID key for Viral Load View. Using '{vl_key}' as index.")

        update_excel_file(FILE_VIRAL_LOAD, df_vl, key_column=vl_key)

    return max_timestamp

def main():
    polling_interval = 300  # Interval check (seconds)
    print("[*] Starting Postgres to Excel Dashboard Sync Pipeline Engine...")

    # Get engine context once
    engine = get_sqlalchemy_engine()
    if not engine:
        print("[-] Fatal: Could not initialize database engine due to missing credentials.")
        return

    while True:
        try:
            # Retrieve where we left off
            last_checkpoint = get_last_synced_timestamp()
            print(f"[*] Checking data mart for changes after checkpoint: {last_checkpoint}")
            
            # Run the updates
            new_checkpoint = check_and_sync_changes(engine, last_checkpoint)
            
            # Keep the tracking tag updated if things progressed
            if new_checkpoint != last_checkpoint:
                save_last_synced_timestamp(new_checkpoint)
                
        except Exception as e:
            print(f"[-] Operational loop failure: {e}")
            
        print(f"[*] Sleeping for {polling_interval} seconds before next delta sweep...")
        time.sleep(polling_interval)

if __name__ == "__main__":
    main()