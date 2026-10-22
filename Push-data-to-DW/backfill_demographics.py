"""
backfill_demographics.py

STANDALONE ONE-OFF SCRIPT
--------------------------
Fixes existing records in marts.lab_lims_statistics that have a NULL
birthdate or gender, by re-checking Hive's fact_lab_request_orders view
and filling in the correct value if Hive now has it.

This does NOT touch your main polling script. Run this once (or any time
you suspect stale nulls have built up), separately from your usual sync job.

Usage:
    python backfill_demographics.py
"""

from pyhive import hive
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()


def log(message):
    """Print with a timestamp and force it to appear immediately (no buffering)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def connect_to_datamart():
    """Connect to the Postgres data mart."""
    log("Connecting to data mart "
        f"(host={os.getenv('DATAMART_HOST')}, db={os.getenv('DATAMART_DB_NAME')})...")
    conn = psycopg2.connect(
        dbname=os.getenv('DATAMART_DB_NAME'),
        user=os.getenv('DATAMART_DB_USERNAME'),
        password=os.getenv('DATAMART_DB_PASSWORD'),
        host=os.getenv('DATAMART_HOST'),
        port=os.getenv('DATAMART_PORT'),
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    log("Data mart connection OK.")
    return conn


def connect_to_hive():
    """Connect to the Hive warehouse."""
    log(f"Connecting to Hive (host={os.getenv('DWH_HOST')}, db={os.getenv('DWH_DATABASE')})...")
    conn = hive.Connection(
        host=os.getenv('DWH_HOST'),
        port=os.getenv('DWH_PORT'),
        username=os.getenv('DWH_USERNAME'),
        password=os.getenv('DWH_PASSWORD'),
        database=os.getenv('DWH_DATABASE'),
        auth=os.getenv('DWH_AUTH_MODE'),
    )
    log("Hive connection OK.")
    return conn


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_person_ids_with_missing_demographics(cur_mart):
    """Find every person_id in the datamart with a null birthdate or gender."""
    log("Querying data mart for person_ids with missing birthdate/gender...")
    cur_mart.execute("""
        SELECT DISTINCT person_id
        FROM marts.lab_lims_statistics
        WHERE birthdate IS NULL OR gender IS NULL
    """)
    ids = [row[0] for row in cur_mart.fetchall()]
    log(f"Query finished. {len(ids)} distinct person_id(s) found.")
    return ids


def fetch_demographics_from_hive(hive_conn, person_ids, chunk_size=500):
    """
    Look up birth_date/gender in Hive for a list of person_ids, in chunks
    (to avoid huge IN clauses). Returns a dict: {person_id: (birth_date, gender)}.
    Only includes people for whom Hive actually has a non-null birth_date.
    """
    found = {}
    cursor = hive_conn.cursor()
    total_chunks = (len(person_ids) + chunk_size - 1) // chunk_size

    for chunk_num, i in enumerate(range(0, len(person_ids), chunk_size), start=1):
        chunk = person_ids[i:i + chunk_size]
        ids_str = "', '".join(chunk)
        query = f"""
            SELECT DISTINCT patient_id, birth_date, gender
            FROM fact_lab_request_orders
            WHERE patient_id IN ('{ids_str}')
            AND birth_date IS NOT NULL
        """
        log(f"Hive lookup: chunk {chunk_num}/{total_chunks} "
            f"({len(chunk)} person_id(s) in this chunk)...")
        cursor.execute(query)
        rows = cursor.fetchall()
        for patient_id, birth_date, gender in rows:
            # If Hive has conflicting rows for the same person, last one wins.
            # (Fine for our purposes -- any non-null value is an improvement.)
            found[patient_id] = (birth_date, gender)

        log(f"Chunk {chunk_num}/{total_chunks} done. "
            f"{len(rows)} matching row(s) returned from Hive. "
            f"Running total with data: {len(found)}.")

    cursor.close()
    return found


def apply_backfill(cur_mart, demographics_by_person):
    """Update the datamart rows that are missing birthdate/gender."""
    updated_count = 0
    total = len(demographics_by_person)
    for idx, (person_id, (birthdate, gender)) in enumerate(demographics_by_person.items(), start=1):
        cur_mart.execute("""
            UPDATE marts.lab_lims_statistics
            SET birthdate = %s, gender = %s
            WHERE person_id = %s AND (birthdate IS NULL OR gender IS NULL)
        """, (birthdate, gender, person_id))
        if cur_mart.rowcount > 0:
            updated_count += cur_mart.rowcount
            log(f"[{idx}/{total}] Backfilled person_id={person_id} "
                f"-> birthdate={birthdate}, gender={gender} "
                f"({cur_mart.rowcount} row(s) updated)")
        else:
            log(f"[{idx}/{total}] person_id={person_id}: no rows needed updating "
                f"(already filled in by the time we got here)")
    return updated_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 70)
    log("Backfill missing demographics in marts.lab_lims_statistics -- STARTING")
    log("=" * 70)

    mart_conn = connect_to_datamart()
    cur_mart = mart_conn.cursor()

    hive_conn = connect_to_hive()

    try:
        log("STEP 1: Finding records with missing birthdate/gender...")
        missing_ids = get_person_ids_with_missing_demographics(cur_mart)

        if not missing_ids:
            log("Nothing to do -- no missing demographics found.")
            return

        log(f"STEP 1 complete. Found {len(missing_ids)} distinct person_id(s) with missing data.")

        log("STEP 2: Checking Hive for current demographic data...")
        demographics_by_person = fetch_demographics_from_hive(hive_conn, missing_ids)

        if not demographics_by_person:
            log("None of the missing person_ids currently have data in Hive either. "
                "Nothing to backfill yet.")
            return

        log(f"STEP 2 complete. Hive has usable data for {len(demographics_by_person)} "
            f"of those {len(missing_ids)} person_id(s).")

        log("STEP 3: Applying backfill to data mart...")
        updated_count = apply_backfill(cur_mart, demographics_by_person)

        log("=" * 70)
        log(f"DONE. {updated_count} row(s) in lab_lims_statistics were backfilled.")
        remaining = len(missing_ids) - len(demographics_by_person)
        if remaining > 0:
            log(f"{remaining} person_id(s) still have no birth_date in Hive "
                f"-- nothing could be done for those (source data is still missing).")
        log("=" * 70)

    finally:
        cur_mart.close()
        mart_conn.close()
        hive_conn.close()
        log("Connections closed.")


if __name__ == "__main__":
    main()