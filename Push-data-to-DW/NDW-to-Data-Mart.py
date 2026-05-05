from pyhive import hive
import psycopg2
import pandas as pd
import time
import math
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Connection to LIMS Data Mart
# conn_mart = psycopg2.connect(dbname="HTSDATA", user="postgres", password="wGMCAE6zFHcyrBmXtus97JPanxvkY4fb", host="127.0.0.1",port= 5431)
# conn_mart = psycopg2.connect(dbname="lims-temporary", user="postgres", password="wGMCAE6zFHcyrBmXtus97JPanxvkY4fb", host="0.0.0.0",port= 5432)
conn_mart = psycopg2.connect(dbname=os.getenv('DATAMART_DB_NAME'),
 user= os.getenv('DATAMART_DB_USERNAME'), password=os.getenv('DATAMART_DB_PASSWORD'),
  host=os.getenv('DATAMART_HOST'),
  port= os.getenv('DATAMART_PORT'))
# conn_mart = psycopg2.connect(dbname="HTS_DB", user="postgres", password="root", host="127.0.0.1",port= 5432)
conn_mart.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
cur_mart = conn_mart.cursor()

# Function to establish a Hive connection
def create_hive_connection(host, port, username, password, database, auth_mode):
    """Creates a connection to the Hive database."""
    try:
        conn = hive.Connection(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            auth=auth_mode
        )
        return conn
    except Exception as e:
        print(f"Error creating connection: {e}")
        return None

# Function to execute a query and fetch results
def execute_hive_query(conn, query):
    """Executes the given query and returns the results."""
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        return results, columns
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        cursor.close()


# Function to get all data from the fact_lab_request_orders
def get_all_patient_data(conn):
 
    """Fetches all data from the fact_lab_request_orders view."""
    query = "SELECT * FROM fact_lab_request_orders where cast(task_authored_on as date) >='2025-02-18' and test_code='ILT0048' "
    return execute_hive_query(conn, query)

# Function to get new or updated patient records (incremental pull)
def get_new_patient_data(conn, last_timestamp):
    """Fetches newly inserted/updated patient data based on last timestamp."""
    # Assuming there is a timestamp column 'last_modified' in the patient table
    query = f"SELECT * FROM fact_lab_request_orders WHERE  test_code= 'ILT0048'  and last_updated > '{last_timestamp}'  order by last_updated asc"
    return execute_hive_query(conn, query)

# Function to close the Hive connection
def close_hive_connection(conn):
    """Closes the connection to the Hive database."""
    try:
        conn.close()
    except Exception as e:
        print(f"Error closing connection: {e}")

# Polling logic to listen for new or updated records
def listen_for_changes(conn, last_timestamp, polling_interval=300):
    """Polls the Hive database for changes at a regular interval."""
    while True:
        try:
            print(f"Polling for changes after {last_timestamp}...")
            print("start here")
            new_data = get_new_patient_data(conn, last_timestamp)
            print("new data", new_data[0])
            
            if new_data[0]:
                print("New/Updated records found:")
                for row in new_data:
                    print(row)
                    # Update the last_timestamp to the latest record's timestamp
                    # Assuming the 'last_modified' column exists and is at index -1
                    last_timestamp = row[-1]  # Update the last timestamp
                    
            else:
                print("No new records found.")
            
            # Wait for the polling interval before checking again
            time.sleep(polling_interval)
        except Exception as e:
            print(f"Error during polling: {e}")
            break  # Exit polling loop if an error occurs

def check_if_snapshot_done():
    cur_mart.execute("SELECT dw_date_created FROM marts.lab_lims_statistics order by dm_date_created desc limit 1") 
    result = cur_mart.fetchone()
    if result:
        return  result[0]
    else:
        return None
    
with open("already_fetched.txt", "w") as f:
    pass

def safe_int(val):
    return None if pd.isna(val) else int(val)


def get_all_patient_data_in_batches(conn, batch_size=50000):
    batch_number = 1
    snapshot_date = check_if_snapshot_done()

    if snapshot_date:
        last_processed_value = snapshot_date #get last date for the snapshot and start from there 
        print(">>> Detected Snapshot done", last_processed_value)
    else:
        last_processed_value = '2025-02-18 00:00:00'  #Initialize last_processed_value to start fetching from
        print(">>> Initial snapshot")

    print("...................................................")

    # Combine all queries into one
    query = f"""
    SELECT 
        COUNT(*) AS total_records, 
        COUNT(DISTINCT encounter_facility_id) AS total_facilities, 
        COUNT(DISTINCT patient_id) AS distinct_people, 
        COUNT(DISTINCT encounter_id) AS distinct_encounters, 
        MAX(last_updated) AS last_update_time
    FROM fact_lab_request_orders where test_code='ILT0048' and last_updated > '{last_processed_value}'
    """
    # Execute the single query
    result = execute_hive_query(conn, query)[0][0]

    # Output the results
    print("Total Number of Records to be fetched:", result[0])
    print("Total Number of Facilities:", result[1])
    print("Number of Distinct People:", result[2])
    print("Number of Distinct Encounters:", result[3])
    print("Last time update:", result[4])

    print("Total Number of Batches ", str(math.ceil(int(result[0])/ 50000)))


    while True:
        print("..................................................")
        print("working on Batch ", batch_number)
        # Query to fetch data in batches

        query = f"SELECT * FROM fact_lab_request_orders where test_code= 'ILT0048' and last_updated > '{last_processed_value}'  order by last_updated asc LIMIT {batch_size}"

        batch_data, columns = execute_hive_query(conn, query)
        
        
        if not batch_data:  # No more data
            print(f"All data fetched. Total batches: {batch_number - 1}")
            break
        
        print(f"Batch {batch_number}: Fetched {len(batch_data)} rows.")
        
        # Append the batch data to the all_data list
        df = pd.DataFrame(batch_data, columns=columns)
        print("Shape of dataframe created ", df.shape)

        # Drop duplicate rows 
        # df['temp_last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
        # df.sort_values(by=['encounter_id', 'temp_last_updated'], inplace=True)
        # df.drop_duplicates(subset=['encounter_id'], keep='last', inplace=True)
        # print("Shape of dataframe after dropping duplicates ", df.shape)

        df['test_type'] =  df['test_type'].str.replace(')','')
        df.rename(columns={'patient_id':'person_id', 
                           'encounter_facility_id':'facility_id_code',
                           'encounter_facility':'facility_name',
                           'birth_date':'birthdate',
                           'last_updated':'date_created',
                           'result':'test_results',
                           'task_authored_on':'shr_date',
                           'task_execution_start_date':'impilo_registration_date',
                           'task_status':'lab_order_status'}, inplace= True)
        
        df['event_date'] = df['shr_date']
        #df.drop(columns=['has_hts_results'], inplace=True)
        print(df.shape)
        
        

        for _, row in df.iterrows():
            encounter_id = row['task_id']
            
            
            my_dict = {}
            with open("already_fetched.txt", "r") as f:
                for line in f:
                    key, value = line.strip().split(":", 1)
                    my_dict[key.strip()] = value.strip()
            
            # Check if a record with the given encounter_id exists
            cur_mart.execute("SELECT 1 FROM marts.lab_lims_statistics WHERE encounter_id = %s", (encounter_id,))
            if cur_mart.fetchone():
                # Update existing record
                print(f"Updating record for encounter_id: {encounter_id}")
                update_query = """
                UPDATE marts.lab_lims_statistics
                SET event_date = %s, dedupe_id = %s, lab_request_number = %s, birthdate = %s, gender = %s, 
                    shr_date = %s, impilo_registration_date = %s, date_sample_taken = %s , lab_order_status = %s, status_reason = %s, note = %s, sample_code = %s, sample_type = %s,
                    test_type = %s,test_code = %s , facility_name = %s, facility_id_code = %s,lab = %s, lab_id = %s, dw_date_created = %s,
                     test_results = %s, date_site_activated= %s, time_facility_to_shr= %s, shr_lab_turn_around_time= %s ,  dm_date_created = NOW(),person_id = %s
                WHERE encounter_id = %s
                """
                cur_mart.execute(update_query, (
                    row['event_date'], row['dedupe_id'], row['lab_request_number'], row['birthdate'], row['gender'], row['shr_date'],
                    row['impilo_registration_date'], row['date_sample_taken'], row['lab_order_status'],
                    row['status_reason'],  row['note'], row['sample_code'],  row['sample_type'],  row['test_type'], row['test_code'],
                    row['facility_name'] ,row['facility_id_code'], row['lab'],  row['lab_id'] ,row['date_created'],  row['test_results'], 
                    row['date_site_activated'], row['time_facility_to_shr'], safe_int(row.get('shr_lab_turn_around_time')),
                     row['person_id'], encounter_id
                ))
            else:
                # Insert new record
                print(f"Inserting new record for encounter_id: {encounter_id}")
                # insert_query = """
                # INSERT INTO marts.dm_hts (
                #     event_date, dedupe_id, birthdate, sex, date_of_hiv_test,
                #     reason_for_hiv_testing, hts_test_result, hts_type, age_at_visit,
                #     first_test_ever_in_life, client_profile, self_identified_gender, 
                #     dw_date_created, dm_date_created, person_id, facility_id_code, received_hiv_test_results,
                #     encounter_id
                # ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s,%s)
                # """
                insert_query = """
                    INSERT INTO marts.lab_lims_statistics (
                    event_date, dedupe_id, lab_request_number, birthdate,  gender, shr_date, impilo_registration_date,
                    date_sample_taken,lab_order_status, status_reason, note, sample_code, sample_type , test_type, test_code,
                    facility_name ,facility_id_code, lab, lab_id, test_results, date_site_activated, time_facility_to_shr, shr_lab_turn_around_time, 
                    dw_date_created, dm_date_created, person_id,
                    encounter_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(),%s,%s)
                """
                cur_mart.execute(insert_query, (
                    row['event_date'], row['dedupe_id'], row['lab_request_number'], row['birthdate'], row['gender'], row['shr_date'],
                    row['impilo_registration_date'], row['date_sample_taken'], row['lab_order_status'],
                    row['status_reason'],  row['note'], row['sample_code'],  row['sample_type'],  row['test_type'],row['test_code'],
                    row['facility_name'],row['facility_id_code'], row['lab'], row['lab_id'], row['test_results'], row['date_site_activated'], row['time_facility_to_shr'], safe_int(row.get('shr_lab_turn_around_time')),
                      row['date_created'], row['person_id'], encounter_id
                ))

        # Update the last_processed_value to the latest timestamp in the batch
        last_processed_value = df['date_created'].max()
        # with open ("already_fetched.txt", "a") as fr:
        #     fr.write(f"{encounter_id}:{row['last_updated']}")
        print("max",last_processed_value)
        print("min", df['date_created'].min())
        batch_number += 1
    return last_processed_value


# Main function 
def main():
 
    # Variable to control initial full load or change listening
    start = "on"  # Set to "on" for initial load or "off" to only listen for changes
    
    # Retry mechanism variables
    retry_attempts = 0
    max_retries = 5000000  # Maximum number of retries before stopping 
    retry_delay = 10  # Delay between retries in seconds
    last_processed_value = '2025-02-18 00:00:00'
    
    while max_retries is None or retry_attempts < max_retries:
        try:
            print(f"Attempting to connect (Attempt {retry_attempts + 1})...")
            
            # Create a connection to Hive
            conn = create_hive_connection(os.getenv('DWH_HOST'), os.getenv('DWH_PORT'), 
            os.getenv('DWH_USERNAME'), os.getenv('DWH_PASSWORD'), os.getenv('DWH_DATABASE'),
             os.getenv('DWH_AUTH_MODE'))
            

            if conn:
                if start == "on":
                    # Fetch all data from the patient table during initial load in batches
                    print("Initial load: Fetching all data from the fact lab-reuqest order view in batches of 50000.")
                    last_processed_value = get_all_patient_data_in_batches(conn, batch_size=50000)
                
                # After initial load, listen for changes (or if start is "off")
                print("Listening for changes in the lab request order view...")
                listen_for_changes(conn, last_processed_value, polling_interval=300)
            
            # Close the connection after completing tasks
            if conn:
                close_hive_connection(conn)
            
            # Reset retry_attempts if everything succeeds
            retry_attempts = 0
            break  # Exit the loop if everything worked successfully
            
        except Exception as e:
            retry_attempts += 1
            print(f"An error occurred: {e}. Retrying in {retry_delay} seconds...")

            # Wait before retrying
            time.sleep(retry_delay)
            
            if max_retries is not None and retry_attempts >= max_retries:
                print(f"Max retries reached ({max_retries}). Exiting.")
                break

# Run the main function
if __name__ == "__main__":
    main()
