CREATE SCHEMA IF NOT EXISTS marts;

CREATE TABLE IF NOT EXISTS marts.dm_lab_request_orders(
    encounter_id VARCHAR(100) PRIMARY KEY,
    event_date DATE,
    dedupe_id VARCHAR(255),
    lab_request_number VARCHAR(100),
    birthdate DATE,
    gender VARCHAR(20),
    shr_date DATE,
    impilo_registration_date DATE,
    date_sample_taken DATE,
    lab_order_status VARCHAR(100),
    status_reason VARCHAR(500),
    note TEXT,
    sample_code VARCHAR(100),
    sample_type VARCHAR(100),
    test_type VARCHAR(100),
    facility_id_code VARCHAR(50),
    lab VARCHAR(100),
    test_results TEXT,
    dw_date_created TIMESTAMP,
    dm_date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    person_id VARCHAR(100)
)
