import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "access_portal.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    user_id TEXT UNIQUE NOT NULL
)
""")

# Insert manually added credentials (only Admin should have access)
users = [
    ("Admin", "admin123"),
]

# Insert data into the table
cursor.executemany("INSERT OR IGNORE INTO users (name, user_id) VALUES (?, ?)", users)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database setup complete. Users added successfully!")
