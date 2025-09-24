import pyhive
from pyhive import hive
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class LabOrderMonitor:
    def __init__(self):
        self.connection = None
        self.facilities = ['ZW090A17', 'ZW090A02', 'ZW090A12', 
                          'ZW090A14', 'ZW090A66', 'ZW090A07']
        
    def connect_to_hive(self):
        """Establish connection to Hive database"""
        try:
            self.connection = hive.connect(
                host='197.221.242.150',
                port=10000,
                username='tjima',
                password='vHYWzTVyygV4Q8tq',
                database='default'
            )
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def check_orders(self, check_date=None):
        """Check orders for a specific date"""
        if check_date is None:
            check_date = datetime.now().strftime('%Y-%m-%d')
        
        query = f"""
        select distinct lab_request_number, task_authored_on, 
               encounter_facility, encounter_facility_id,  
               task_status, date_sample_taken
        from fact_lab_request_orders
        where encounter_facility_id in {tuple(self.facilities)}
        and cast(task_authored_on as date) = '{check_date}'
        and lab = 'MPILO' 
        and test_type like '%Viral Load%'
        """
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Query failed: {e}")
            return pd.DataFrame()
    
    def generate_report(self, df, check_date):
        """Generate a summary report"""
        if df.empty:
            return "No orders found for the specified date."
        
        report = f"Lab Order Report for {check_date}\n"
        report += "=" * 50 + "\n\n"
        
        # Facilities with orders
        facilities_with_orders = df['encounter_facility_id'].unique()
        facilities_without_orders = [f for f in self.facilities 
                                   if f not in facilities_with_orders]
        
        report += "Facilities WITH orders:\n"
        for facility in facilities_with_orders:
            facility_orders = df[df['encounter_facility_id'] == facility]
            report += f"  - {facility}: {len(facility_orders)} orders\n"
        
        report += "\nFacilities WITHOUT orders:\n"
        for facility in facilities_without_orders:
            report += f"  - {facility}: No orders\n"
        
        return report
    
    # def send_email(self, report, recipients):
    #     """Send report via email"""
    #     msg = MIMEMultipart()
    #     msg['Subject'] = 'Daily Lab Order Report'
    #     msg['From'] = 'your_email@example.com'
    #     msg['To'] = ', '.join(recipients)
        
    #     msg.attach(MIMEText(report, 'plain'))
        
    #     try:
    #         with smtplib.SMTP('smtp_server', 587) as server:
    #             server.starttls()
    #             server.login('username', 'password')
    #             server.send_message(msg)
    #         print("Email sent successfully")
    #     except Exception as e:
    #         print(f"Email failed: {e}")
    
    def daily_check(self):
        """Run daily check"""
        if self.connect_to_hive():
            today = datetime.now().strftime('%Y-%m-%d')
            df = self.check_orders(today)
            report = self.generate_report(df, today)
            
            # Print report
            print(report)
            
            # Send email (uncomment to enable)
            # self.send_email(report, ['recipient1@example.com', 'recipient2@example.com'])
            
            self.connection.close()

# Usage
monitor = LabOrderMonitor()

# Run once
monitor.daily_check()

# Or schedule daily at 8 AM
schedule.every().day.at("08:00").do(monitor.daily_check)

while True:
    schedule.run_pending()
    time.sleep(60)
    
    
    
    
    
