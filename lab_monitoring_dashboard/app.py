import os
from flask import Flask, render_template, request, jsonify
import pyhive
from pyhive import hive
import pandas as pd
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Database configuration
HIVE_HOST = os.environ.get('HIVE_HOST')
HIVE_PORT = os.environ.get('HIVE_PORT')
HIVE_USERNAME = os.environ.get('HIVE_USER')
HIVE_PASSWORD = os.environ.get('HIVE_PASSWORD')
HIVE_DATABASE = os.environ.get('HIVE_DATABASE')

def get_hive_connection():
    
    """Establish connection to Hive database"""
    try:
        conn = hive.Connection(
            host=HIVE_HOST,
            port=HIVE_PORT,
            username=HIVE_USERNAME,
            password=HIVE_PASSWORD,
            database=HIVE_DATABASE,
            auth='LDAP'  
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
    """Return facilities that sent data on a specific date, with stats"""
    try:
        data = request.get_json()
        date = data.get('date')
        selected_facilities = data.get('facilities', [])
        if not date:
            return jsonify({'error': 'Missing date parameter'}), 400

        facility_map = {
            'ZW090A17': 'Nketa',
            'ZW090A02': 'Tshabalala',
            'ZW090A12': 'Princess Margaret',
            'ZW090A14': 'Dr. Shennan',
            'ZW090A66': 'Mahatshula',
            'ZW090A07': 'Magwegwe'
        }
        # Use selected facilities if provided, else all
        facilities = selected_facilities if selected_facilities else list(facility_map.keys())
        conn = get_hive_connection()
        if not conn:
            return jsonify({'error': 'Failed to connect to Hive'}), 500

        # Orders for selected date
        query_today = f"""
        SELECT encounter_facility_id, COUNT(DISTINCT lab_request_number) AS orders_today
        FROM fact_lab_request_orders
        WHERE encounter_facility_id IN ({", ".join([f"'{f}'" for f in facilities])})
            AND cast(task_authored_on as date) = '{date}'
            AND lab = 'MPILO'
            AND test_type LIKE '%Viral Load%'
        GROUP BY encounter_facility_id
        """

        # Total requests from 2025-02-18 up to now
        query_total = f"""
        SELECT encounter_facility_id, COUNT(DISTINCT lab_request_number) AS total_orders
        FROM fact_lab_request_orders
        WHERE encounter_facility_id IN ({", ".join([f"'{f}'" for f in facilities])})
            AND cast(task_authored_on as date) >= '2025-02-18'
            AND lab = 'MPILO'
            AND test_type LIKE '%Viral Load%'
        GROUP BY encounter_facility_id
        """

        # Last request for each facility
        query_last = f"""
        SELECT encounter_facility_id, MAX(task_authored_on) AS last_request
        FROM fact_lab_request_orders
        WHERE encounter_facility_id IN ({", ".join([f"'{f}'" for f in facilities])})
            AND lab = 'MPILO'
            AND test_type LIKE '%Viral Load%'
        GROUP BY encounter_facility_id
        """

        df_today = pd.read_sql_query(query_today, conn)
        df_total = pd.read_sql_query(query_total, conn)
        df_last = pd.read_sql_query(query_last, conn)

        results = []
        for fid in facilities:
            name = facility_map.get(fid, fid)
            orders_today = int(df_today[df_today['encounter_facility_id'] == fid]['orders_today'].values[0]) if not df_today[df_today['encounter_facility_id'] == fid].empty else 0
            total_orders = int(df_total[df_total['encounter_facility_id'] == fid]['total_orders'].values[0]) if not df_total[df_total['encounter_facility_id'] == fid].empty else 0
            last_request = df_last[df_last['encounter_facility_id'] == fid]['last_request'].values[0] if not df_last[df_last['encounter_facility_id'] == fid].empty else None
            results.append({
                'facility_id': fid,
                'facility_name': name,
                'orders_today': orders_today,
                'total_orders': total_orders,
                'last_request': last_request
            })
        return jsonify({'facilities': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/facility_statuses', methods=['POST'])
def facility_statuses():
    """Return status counts for each facility in a date range"""
    try:
        data = request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        facilities = data.get('facilities', [])
        facility_map = {
            'ZW090A17': 'Nketa',
            'ZW090A02': 'Tshabalala',
            'ZW090A12': 'Princess Margaret',
            'ZW090A14': 'Dr. Shennan',
            'ZW090A66': 'Mahatshula',
            'ZW090A07': 'Magwegwe'
        }
        if not start_date or not end_date or not facilities:
            return jsonify({'error': 'Missing parameters'}), 400

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
            AND cast(task_authored_on as date) >= '{start_date}'
            AND cast(task_authored_on as date) <= '{end_date}'
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
     app.run(host='0.0.0.0', port=5000, debug=True)


