import sqlite3
import sqlite_vec
import os

db_path = 'embedding.db' # Based on my previous logs
if not os.path.exists(db_path):
    # Try memory or default
    db_path = ':memory:'

conn = sqlite3.connect(db_path)
conn.enable_load_extension(True)
sqlite_vec.load(conn)

cursor = conn.cursor()
# Create a dummy table to see what it has
try:
    cursor.execute("CREATE VIRTUAL TABLE test_vec USING vec0(contents TEXT, embedding float[3])")
    cursor.execute("INSERT INTO test_vec(contents, embedding) VALUES (?, ?)", ("test", b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'))
    
    print("Columns in test_vec:")
    cursor.execute("PRAGMA table_info(test_vec)")
    for row in cursor.fetchall():
        print(row)
        
    print("\nAttempting to select rowid from test_vec:")
    cursor.execute("SELECT rowid, contents FROM test_vec")
    print(cursor.fetchone())
    
    cursor.execute("CREATE VIRTUAL TABLE test_fts USING fts5(contents)")
    print("\nColumns in test_fts:")
    cursor.execute("PRAGMA table_info(test_fts)")
    for row in cursor.fetchall():
        print(row)
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
