from pyhive import hive
import psycopg2
import pandas as pd
import time
import math
import logging
from datetime import datetime
from dotenv import load_dotenv
import os
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataPipeline:
    def __init__(self):
        self.conn_mart = None
        self.cur_mart = None
        self.hive_conn = None
        self.stats = {
            'new_records': 0,
            'updated_records': 0,
            'errors': 0
        }
        self.initialize_db_connection()
    
    def initialize_db_connection(self):
        """Initialize PostgreSQL connection"""
        try:
            self.conn_mart = psycopg2.connect(
                dbname=os.getenv('DATAMART_DB_NAME'),
                user=os.getenv('DATAMART_DB_USERNAME'),
                password=os.getenv('DATAMART_DB_PASSWORD'),
                host=os.getenv('DATAMART_HOST'),
                port=os.getenv('DATAMART_PORT')
            )
            # self.conn_mart = psycopg2.connect(dbname="LIMSDATA", user="postgres", password="wGMCAE6zFHcyrBmXtus97JPanxvkY4fb", host="ehr-lab-datamart-postgres", port= 5432)
            self.conn_mart.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            self.cur_mart = self.conn_mart.cursor()
            logger.info("PostgreSQL connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def create_hive_connection(self):
        """Creates a connection to the Hive database"""
        try:
            conn = hive.Connection(
                host=os.getenv('DWH_HOST'),
                port=int(os.getenv('DWH_PORT')),
                username=os.getenv('DWH_USERNAME'),
                password=os.getenv('DWH_PASSWORD'),
                database=os.getenv('DWH_DATABASE'),
                auth=os.getenv('DWH_AUTH_MODE')
            )
            logger.info("Hive connection established successfully")
            return conn
        except Exception as e:
            logger.error(f"Error creating Hive connection: {e}")
            return None

    def execute_hive_query(self, conn, query):
        """Executes the given query and returns the results"""
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return results, columns
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None, None

    def check_if_snapshot_done(self):
        """Check the last snapshot date"""
        try:
            self.cur_mart.execute(
                "SELECT dw_date_created FROM marts.lab_lims_statistics "
                "ORDER BY dm_date_created DESC LIMIT 1"
            )
            result = self.cur_mart.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error checking snapshot: {e}")
            return None

    def get_data_statistics(self, conn, last_processed_value):
        """Get statistics about data to be fetched"""
        query = f"""
        SELECT 
            COUNT(*) AS total_records, 
            COUNT(DISTINCT encounter_facility_id) AS total_facilities, 
            COUNT(DISTINCT patient_id) AS distinct_people, 
            COUNT(DISTINCT encounter_id) AS distinct_encounters, 
            MAX(last_updated) AS last_update_time
        FROM fact_lab_request_orders 
        WHERE test_code='ILT0048' AND last_updated > '{last_processed_value}'
        """
        result, _ = self.execute_hive_query(conn, query)
        if result and result[0]:
            return result[0]
        return (0, 0, 0, 0, None)

    def process_batch(self, batch_data, columns):
        """Process a batch of data and insert/update in PostgreSQL"""
        df = pd.DataFrame(batch_data, columns=columns)
        
        # Data transformations
        df['test_type'] = df['test_type'].str.replace(')', '')
        df.rename(columns={
            'patient_id': 'person_id',
            'encounter_facility_id': 'facility_id_code',
            'encounter_facility': 'facility_name',
            'birth_date': 'birthdate',
            'last_updated': 'date_created',
            'result': 'test_results',
            'task_authored_on': 'shr_date',
            'task_execution_start_date': 'impilo_registration_date',
            'task_status': 'lab_order_status'
        }, inplace=True)
        
        df['event_date'] = df['shr_date']
        
        batch_new = 0
        batch_updated = 0
        batch_errors = 0
        
        for _, row in df.iterrows():
            encounter_id = row['task_id']
            
            try:
                # Check if record exists
                self.cur_mart.execute(
                    "SELECT 1 FROM marts.lab_lims_statistics WHERE encounter_id = %s",
                    (encounter_id,)
                )
                
                if self.cur_mart.fetchone():
                    # Update existing record
                    update_query = """
                    UPDATE marts.lab_lims_statistics
                    SET event_date = %s, dedupe_id = %s, lab_request_number = %s, 
                        birthdate = %s, gender = %s, shr_date = %s, 
                        impilo_registration_date = %s, date_sample_taken = %s, 
                        lab_order_status = %s, status_reason = %s, note = %s, 
                        sample_code = %s, sample_type = %s, test_type = %s, 
                        test_code = %s, facility_name = %s, facility_id_code = %s, 
                        lab = %s, lab_id = %s, dw_date_created = %s, 
                        test_results = %s, dm_date_created = NOW(), person_id = %s
                    WHERE encounter_id = %s
                    """
                    self.cur_mart.execute(update_query, (
                        row['event_date'], row['dedupe_id'], row['lab_request_number'],
                        row['birthdate'], row['gender'], row['shr_date'],
                        row['impilo_registration_date'], row['date_sample_taken'],
                        row['lab_order_status'], row['status_reason'], row['note'],
                        row['sample_code'], row['sample_type'], row['test_type'],
                        row['test_code'], row['facility_name'], row['facility_id_code'],
                        row['lab'], row['lab_id'], row['date_created'],
                        row['test_results'], row['person_id'], encounter_id
                    ))
                    batch_updated += 1
                else:
                    # Insert new record
                    insert_query = """
                    INSERT INTO marts.lab_lims_statistics (
                        event_date, dedupe_id, lab_request_number, birthdate, gender,
                        shr_date, impilo_registration_date, date_sample_taken,
                        lab_order_status, status_reason, note, sample_code,
                        sample_type, test_type, test_code, facility_name,
                        facility_id_code, lab, lab_id, test_results,
                        dw_date_created, dm_date_created, person_id, encounter_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                              %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s)
                    """
                    self.cur_mart.execute(insert_query, (
                        row['event_date'], row['dedupe_id'], row['lab_request_number'],
                        row['birthdate'], row['gender'], row['shr_date'],
                        row['impilo_registration_date'], row['date_sample_taken'],
                        row['lab_order_status'], row['status_reason'], row['note'],
                        row['sample_code'], row['sample_type'], row['test_type'],
                        row['test_code'], row['facility_name'], row['facility_id_code'],
                        row['lab'], row['lab_id'], row['test_results'],
                        row['date_created'], row['person_id'], encounter_id
                    ))
                    batch_new += 1
                    
            except Exception as e:
                logger.error(f"Error processing encounter_id {encounter_id}: {e}")
                batch_errors += 1
        
        return batch_new, batch_updated, batch_errors, df['date_created'].max()

    def run_pipeline(self, batch_size=10000):
        """Main pipeline execution"""
        logger.info("=" * 60)
        logger.info("Starting pipeline execution")
        logger.info("=" * 60)
        
        # Reset statistics
        self.stats = {'new_records': 0, 'updated_records': 0, 'errors': 0}
        
        try:
            # Create Hive connection
            self.hive_conn = self.create_hive_connection()
            if not self.hive_conn:
                logger.error("Failed to establish Hive connection")
                return
            
            # Get last processed timestamp
            snapshot_date = self.check_if_snapshot_done()
            if snapshot_date:
                last_processed_value = snapshot_date
                logger.info(f"Resuming from last snapshot: {last_processed_value}")
            else:
                last_processed_value = '2025-02-18 00:00:00'
                logger.info("Starting initial snapshot")
            
            # Get data statistics
            stats = self.get_data_statistics(self.hive_conn, last_processed_value)
            total_records = stats[0]
            total_batches = math.ceil(total_records / batch_size) if total_records > 0 else 0
            
            logger.info(f"Total records to process: {total_records}")
            logger.info(f"Total facilities: {stats[1]}")
            logger.info(f"Distinct patients: {stats[2]}")
            logger.info(f"Distinct encounters: {stats[3]}")
            logger.info(f"Total batches: {total_batches}")
            
            if total_records == 0:
                logger.info("No new records to process")
                return
            
            # Process batches
            batch_number = 1
            while True:
                logger.info(f"Processing batch {batch_number}/{total_batches}")
                
                query = f"""
                SELECT * FROM fact_lab_request_orders 
                WHERE test_code='ILT0048' AND last_updated > '{last_processed_value}'  
                ORDER BY last_updated ASC 
                LIMIT {batch_size}
                """
                
                batch_data, columns = self.execute_hive_query(self.hive_conn, query)
                
                if not batch_data:
                    logger.info(f"All data processed. Total batches: {batch_number - 1}")
                    break
                
                logger.info(f"Batch {batch_number}: Processing {len(batch_data)} rows")
                
                # Process the batch
                new, updated, errors, max_timestamp = self.process_batch(batch_data, columns)
                
                self.stats['new_records'] += new
                self.stats['updated_records'] += updated
                self.stats['errors'] += errors
                
                logger.info(f"Batch {batch_number} complete: {new} new, {updated} updated, {errors} errors")
                
                last_processed_value = max_timestamp
                batch_number += 1
            
            # Final summary
            logger.info("=" * 60)
            logger.info("Pipeline execution completed successfully")
            logger.info(f"Total new records: {self.stats['new_records']}")
            logger.info(f"Total updated records: {self.stats['updated_records']}")
            logger.info(f"Total errors: {self.stats['errors']}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.stats['errors'] += 1
        
        finally:
            if self.hive_conn:
                try:
                    self.hive_conn.close()
                    logger.info("Hive connection closed")
                except Exception as e:
                    logger.error(f"Error closing Hive connection: {e}")

    def close(self):
        """Close database connections"""
        if self.cur_mart:
            self.cur_mart.close()
        if self.conn_mart:
            self.conn_mart.close()
        logger.info("PostgreSQL connection closed")


def main():
    """Main function to run pipeline on schedule"""
    logger.info("Initializing scheduled pipeline")
    
    pipeline = DataPipeline()
    
    # Run immediately on start
    logger.info("Running initial pipeline execution")
    pipeline.run_pipeline(batch_size=10000)
    
    # Schedule to run every 30 minutes
    schedule.every(60).minutes.do(pipeline.run_pipeline, batch_size=10000)
    
    logger.info("Pipeline scheduled to run every 60 minutes")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()