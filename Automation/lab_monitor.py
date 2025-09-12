# simple_monitor.py
import subprocess
from datetime import datetime
import sys

def run_hive_query(date):
    """Run the Hive query using beeline command line"""
    facilities = "'ZW090A17', 'ZW090A02', 'ZW090A12', 'ZW090A14', 'ZW090A66', 'ZW090A07'"
    
    query = f"""
    SELECT distinct lab_request_number, task_authored_on, 
           encounter_facility, encounter_facility_id,  
           task_status, date_sample_taken
    FROM fact_lab_request_orders
    WHERE encounter_facility_id in ({facilities})
    AND cast(task_authored_on as date) = '{date}'
    AND lab = 'MPILO' 
    AND test_type like '%Viral Load%'
    """
    
    # Format query for beeline
    beeline_cmd = f'beeline -u "jdbc:hive2://your_hive_host:10000" -e "{query}"'
    
    try:
        result = subprocess.run(beeline_cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def main():
    if len(sys.argv) > 1:
        check_date = sys.argv[1]
    else:
        check_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Checking orders for: {check_date}")
    result = run_hive_query(check_date)
    print(result)

if __name__ == "__main__":
    main()