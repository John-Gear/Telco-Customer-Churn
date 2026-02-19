import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "sql" / "Telco-Customer-Churn.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def read_sql(qerry: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(qerry, conn)
    conn.close()
    return df