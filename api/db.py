# src/api/db.py

import os
import psycopg2
import psycopg2.extras
from datetime import datetime

# If DATABASE_URL_PROD exists (Cloud Run), use it.
# Otherwise, fallback to DATABASE_URL (Docker Compose or local).
DB_URL = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URL_PROD")

def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def save_audit_record(record):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO audit_log (
            timestamp,
            transaction_id,
            transaction_json,
            fraud_probability,
            top_factors,
            explanation,
            decision,
            audit_log
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id;
    """, (
        datetime.utcnow(),
        record["transaction_id"],
        psycopg2.extras.Json(record["transaction"]),
        record["fraud_probability"],
        psycopg2.extras.Json(record["top_factors"]),
        record["explanation"],
        record["decision"],
        psycopg2.extras.Json(record["audit_log"])
    ))

    new_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return new_id


def get_audit_record(record_id: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM audit_log WHERE id = %s;", (record_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()
    return row
