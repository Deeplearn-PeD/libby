#!/usr/bin/env python3
"""
Migration script to transfer embeddings from DuckDB to PostgreSQL.

Usage:
    python scripts/migrate_duckdb_to_postgres.py --duckdb-path /path/to/embeddings.duckdb --postgres-url postgresql://user:pass@host:port/dbname
"""

import argparse
import sys
from pathlib import Path

import duckdb
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from tqdm import tqdm

from libbydbot.brain.embed import Base, Embedding


def migrate_embeddings(duckdb_path: str, postgres_url: str, batch_size: int = 1000):
    """
    Migrate embeddings from DuckDB to PostgreSQL.

    :param duckdb_path: Path to DuckDB database file
    :param postgres_url: PostgreSQL connection URL
    :param batch_size: Number of records to process in each batch
    """
    if not Path(duckdb_path).exists():
        logger.error(f"DuckDB file not found: {duckdb_path}")
        sys.exit(1)

    logger.info(f"Connecting to DuckDB: {duckdb_path}")
    duck_conn = duckdb.connect(duckdb_path, read_only=True)

    logger.info(
        f"Connecting to PostgreSQL: {postgres_url.split('@')[1] if '@' in postgres_url else postgres_url}"
    )
    pg_engine = create_engine(postgres_url)

    logger.info("Checking if pgvector extension is installed...")
    with pg_engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            logger.info("pgvector extension ready")
        except Exception as e:
            logger.error(f"Failed to create pgvector extension: {e}")
            logger.error("Make sure you're using the pgvector/pgvector Docker image")
            sys.exit(1)

    logger.info("Creating tables in PostgreSQL if they don't exist...")
    Base.metadata.create_all(pg_engine, checkfirst=True)

    tables = duck_conn.execute("SHOW TABLES").fetchall()
    table_names = [table[0] for table in tables]

    if "embedding_duckdb" not in table_names:
        logger.error(
            f"Table 'embedding_duckdb' not found in DuckDB. Available tables: {table_names}"
        )
        duck_conn.close()
        sys.exit(1)

    count_result = duck_conn.execute("SELECT COUNT(*) FROM embedding_duckdb").fetchone()
    total_records = count_result[0]

    if total_records == 0:
        logger.info("No records to migrate. DuckDB database is empty.")
        duck_conn.close()
        return

    logger.info(f"Found {total_records} records to migrate")

    columns = [
        "collection_name",
        "doc_name",
        "page_number",
        "doc_hash",
        "document",
        "embedding_model",
        "embedding",
    ]

    offset = 0
    migrated_count = 0
    skipped_count = 0

    with Session(pg_engine) as pg_session:
        with tqdm(total=total_records, desc="Migrating embeddings") as pbar:
            while offset < total_records:
                batch_query = f"""
                    SELECT {", ".join(columns)}
                    FROM embedding_duckdb
                    ORDER BY id
                    LIMIT {batch_size} OFFSET {offset}
                """

                batch = duck_conn.execute(batch_query).fetchall()

                if not batch:
                    break

                for row in batch:
                    (
                        collection_name,
                        doc_name,
                        page_number,
                        doc_hash,
                        document,
                        embedding_model,
                        embedding,
                    ) = row

                    existing = pg_session.execute(
                        text("SELECT id FROM embedding WHERE doc_hash = :hash"),
                        {"hash": doc_hash},
                    ).first()

                    if existing:
                        skipped_count += 1
                        continue

                    embedding_record = Embedding(
                        collection_name=collection_name,
                        doc_name=doc_name,
                        page_number=page_number,
                        doc_hash=doc_hash,
                        document=document,
                        embedding_model=embedding_model or "embeddinggemma",
                        embedding=embedding,
                    )

                    pg_session.add(embedding_record)
                    migrated_count += 1

                try:
                    pg_session.commit()
                    pbar.update(len(batch))
                except Exception as e:
                    logger.error(f"Error committing batch: {e}")
                    pg_session.rollback()
                    raise

                offset += batch_size

    duck_conn.close()

    logger.success(f"Migration completed successfully!")
    logger.info(f"  Total records: {total_records}")
    logger.info(f"  Migrated: {migrated_count}")
    logger.info(f"  Skipped (duplicates): {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate embeddings from DuckDB to PostgreSQL"
    )
    parser.add_argument(
        "--duckdb-path", required=True, help="Path to DuckDB database file"
    )
    parser.add_argument(
        "--postgres-url",
        required=True,
        help="PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/dbname)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to process in each batch (default: 1000)",
    )

    args = parser.parse_args()

    try:
        migrate_embeddings(args.duckdb_path, args.postgres_url, args.batch_size)
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
