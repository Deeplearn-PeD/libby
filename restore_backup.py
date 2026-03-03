#!/usr/bin/env python3
"""
Restore the embedding_duckdb table from the backup table.
"""

import duckdb

print("Connecting to database...")
conn = duckdb.connect("data/embeddings.duckdb")

# Check current state
tables = conn.sql(
    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
).fetchall()
print(f"Tables found: {[t[0] for t in tables]}")

# Check counts
try:
    main_count = conn.sql("SELECT COUNT(*) FROM embedding_duckdb").fetchone()[0]
    print(f"Main table (embedding_duckdb) count: {main_count}")
except Exception as e:
    print(f"Main table error: {e}")
    main_count = 0

try:
    backup_count = conn.sql(
        "SELECT COUNT(*) FROM embedding_duckdb_backup_reembed"
    ).fetchone()[0]
    print(f"Backup table (embedding_duckdb_backup_reembed) count: {backup_count}")
except Exception as e:
    print(f"Backup table error: {e}")
    backup_count = 0

if backup_count == 0:
    print("\nERROR: Backup table is empty or doesn't exist!")
    conn.close()
    exit(1)

if main_count > 0:
    print(f"\nWARNING: Main table already has {main_count} records.")
    response = input("Do you want to restore anyway? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        conn.close()
        exit(0)

print("\nRestoring from backup...")

# Get schema from backup
schema = conn.sql("DESCRIBE embedding_duckdb_backup_reembed").fetchall()
print("Backup schema:")
for col in schema:
    print(f"  {col[0]}: {col[1]}")

# Drop the empty main table
conn.sql("DROP TABLE IF EXISTS embedding_duckdb")
print("Dropped empty main table")

# Rename backup to main
conn.sql("ALTER TABLE embedding_duckdb_backup_reembed RENAME TO embedding_duckdb")
print("Renamed backup table to main table")

# Verify restoration
restored_count = conn.sql("SELECT COUNT(*) FROM embedding_duckdb").fetchone()[0]
print(f"\nRestored {restored_count} records")

# Recreate FTS index
print("\nRecreating FTS index...")
try:
    conn.sql("INSTALL fts;")
    conn.sql("LOAD fts;")
    conn.sql("PRAGMA create_fts_index('embedding_duckdb', 'id', 'document');")
    print("FTS index created successfully")
except Exception as e:
    print(f"Warning: Could not create FTS index: {e}")

conn.close()
print("\nRestoration complete!")
