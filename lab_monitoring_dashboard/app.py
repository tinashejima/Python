from flask import Flask, render_template, request, jsonify
import pyhive
from pyhive import hive
import pandas as pd
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Database configuration (update with your actual credentials)
HIVE_HOST = '197.221.242.150'
HIVE_PORT = 17251
HIVE_USERNAME = 'tjima'
HIVE_PASSWORD = 'vHYWzTVyygV4Q8tq'
HIVE_DATABASE = 'default'

def get_hive_connection():
    
    """Establish connection to Hive database"""
    try:
        conn = hive.Connection(
            host=HIVE_HOST,
            port=HIVE_PORT,
            username=HIVE_USERNAME,
            password=HIVE_PASSWORD,
            database=HIVE_DATABASE,
            auth='LDAP'  # or other authentication method
        )
        return conn
    except Exception as e:
        print(f"Error connecting to Hive: {str(e)}")
        return None

def execute_query(facilities, start_date, end_date):
    """Execute the Hive query with provided parameters"""
    conn = get_hive_connection()
    if not conn:
        return None
    
    try:
        # Format facilities for SQL query
        facilities_str = ", ".join([f"'{f}'" for f in facilities])
        
        # Build the query
        query = f"""
        select distinct 
            lab_request_number, 
            task_authored_on,
            encounter_facility, 
            encounter_facility_id, 
            task_status, 
            date_sample_taken 
        from fact_lab_request_orders 
        where encounter_facility_id in ({facilities_str})
            and cast(task_authored_on as date) >= '{start_date}'
            and cast(task_authored_on as date) <= '{end_date}'
            and lab = 'MPILO' 
            and test_type like '%Viral Load%'
        """
        
        # Execute query and fetch results
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

@app.route('/')
def index():
    """Main page with form and results"""
    # Default date range (last 7 days)
    default_end = datetime.now().strftime('%Y-%m-%d')
    default_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Predefined facility list
    facilities = [
        {'id': 'ZW090A17', 'name': 'Nketa'},
        {'id': 'ZW090A02', 'name': 'Tshabalala'},
        {'id': 'ZW090A12', 'name': 'Princess Margaret'},
        {'id': 'ZW090A14', 'name': 'Dr. Shennan'},
        {'id': 'ZW090A66', 'name': 'Mahatshula'},
        {'id': 'ZW090A07', 'name': 'Magwegwe'}
    ]
    
    return render_template('index.html', 
                         facilities=facilities,
                         default_start=default_start,
                         default_end=default_end)


@app.route('/active_facilities', methods=['POST'])
def active_facilities():
    """Return facilities that sent data on a specific date"""
    try:
        data = request.get_json()
        date = data.get('date')
        if not date:
            return jsonify({'error': 'Missing date parameter'}), 400

        conn = get_hive_connection()
        if not conn:
            return jsonify({'error': 'Failed to connect to Hive'}), 500

        query = f"""
        SELECT DISTINCT encounter_facility_id, encounter_facility
        FROM fact_lab_request_orders
        WHERE cast(task_authored_on as date) = '{date}'
            AND lab = 'MPILO'
            AND test_type LIKE '%Viral Load%'
        """

        df = pd.read_sql_query(query, conn)
        facilities = df.to_dict('records')
        return jsonify({'facilities': facilities, 'count': len(facilities)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/facility_statuses', methods=['POST'])
def facility_statuses():
    """Return status counts for each facility on a specific date"""
    try:
        data = request.get_json()
        date = data.get('date')
        facilities = data.get('facilities', [])
        # You need to also get facility names from your config
        facility_map = {
            'ZW090A17': 'Nketa',
            'ZW090A02': 'Tshabalala',
            'ZW090A12': 'Princess Margaret',
            'ZW090A14': 'Dr. Shennan',
            'ZW090A66': 'Mahatshula',
            'ZW090A07': 'Magwegwe'
        }
        if not date or not facilities:
            return jsonify({'error': 'Missing date or facilities parameter'}), 400

        conn = get_hive_connection()
        if not conn:
            return jsonify({'error': 'Failed to connect to Hive'}), 500

        facilities_str = ", ".join([f"'{f}'" for f in facilities])
        query = f"""
        SELECT 
            encounter_facility_id, 
            encounter_facility,
            task_status,
            COUNT(DISTINCT lab_request_number) AS order_count
        FROM fact_lab_request_orders
        WHERE encounter_facility_id IN ({facilities_str})
            AND cast(task_authored_on as date) = '{date}'
            AND lab = 'MPILO'
            AND test_type LIKE '%Viral Load%'
        GROUP BY encounter_facility_id, encounter_facility, task_status
        """

        import pandas as pd
        df = pd.read_sql_query(query, conn)
        result = {}
        for facility in facilities:
            facility_rows = df[df['encounter_facility_id'] == facility]
            if facility_rows.empty:
                # Use the name from your map
                result[facility] = {
                    'name': facility_map.get(facility, facility),
                    'statuses': {},
                    'no_orders': True
                }
            else:
                statuses = {row['task_status']: int(row['order_count']) for _, row in facility_rows.iterrows()}
                name = facility_rows.iloc[0]['encounter_facility']
                result[facility] = {
                    'name': name,
                    'statuses': statuses,
                    'no_orders': False
                }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)


