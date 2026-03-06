#!/usr/bin/env python3
"""
Enhanced Migration script to transfer embeddings from DuckDB to PostgreSQL.

Features:
- Pre-flight checks and validation
- Dry-run mode to preview migration
- Progress tracking with time estimation
- Automatic duplicate detection
- Data integrity verification
- Detailed migration report
- Resume capability
- Dimension mismatch detection and re-embedding

Usage:
    # Preview migration (dry-run)
    python scripts/migrate_duckdb_to_postgres.py --dry-run --duckdb-path ./embeddings.duckdb --postgres-url postgresql://user:pass@host:port/dbname

    # Run actual migration
    python scripts/migrate_duckdb_to_postgres.py --duckdb-path ./embeddings.duckdb --postgres-url postgresql://user:pass@host:port/dbname

    # Resume interrupted migration
    python scripts/migrate_duckdb_to_postgres.py --resume --duckdb-path ./embeddings.duckdb --postgres-url postgresql://user:pass@host:port/dbname

    # Re-embed when dimension mismatch
    python scripts/migrate_duckdb_to_postgres.py --re-embed --duckdb-path ./embeddings.duckdb --postgres-url postgresql://user:pass@host:port/dbname
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, quote

import duckdb
from loguru import logger
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from tqdm import tqdm

from libbydbot.brain.embed import Base, Embedding


def get_embedding_dimension_duckdb(duck_conn, embedding_table):
    """Get the embedding dimension from DuckDB."""
    try:
        result = duck_conn.execute(
            f"SELECT embedding FROM {embedding_table} LIMIT 1"
        ).fetchone()
        if result and result[0]:
            return len(result[0])
        return None
    except Exception as e:
        logger.warning(f"Could not determine embedding dimension from DuckDB: {e}")
        return None


def get_expected_postgres_dimension():
    """Get the expected embedding dimension for PostgreSQL (1024)."""
    # This matches the Vector(1024) column in the Embedding model
    return 1024


def check_embedding_dimension(duck_conn, embedding_table, expected_dim=1024):
    """Check if embedding dimensions match between DuckDB and PostgreSQL."""
    duckdb_dim = get_embedding_dimension_duckdb(duck_conn, embedding_table)

    if duckdb_dim is None:
        logger.warning("Could not determine DuckDB embedding dimension")
        return False, None

    logger.info(f"DuckDB embedding dimension: {duckdb_dim}")
    logger.info(f"PostgreSQL expected dimension: {expected_dim}")

    if duckdb_dim != expected_dim:
        logger.warning(
            f"Dimension mismatch detected: DuckDB={duckdb_dim}, PostgreSQL={expected_dim}"
        )
        return False, duckdb_dim

    return True, duckdb_dim


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.start_time = None
        self.total_records = 0
        self.migrated_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.batch_count = 0

    def start(self):
        self.start_time = time.time()

    def elapsed_time(self):
        if self.start_time:
            return time.time() - self.start_time
        return 0

    def estimate_remaining(self, processed):
        if processed > 0 and self.start_time:
            elapsed = self.elapsed_time()
            rate = processed / elapsed
            remaining = self.total_records - processed
            return remaining / rate if rate > 0 else 0
        return 0

    def summary(self):
        elapsed = self.elapsed_time()
        return f"""
{"=" * 60}
Migration Summary
{"=" * 60}
Total records processed:  {self.total_records}
Successfully migrated:    {self.migrated_count}
Skipped (duplicates):     {self.skipped_count}
Errors:                   {self.error_count}
Batches processed:        {self.batch_count}
Total time:               {elapsed:.2f} seconds
Average rate:             {self.total_records / elapsed if elapsed > 0 else 0:.2f} records/sec
{"=" * 60}
"""


def check_prerequisites(duckdb_path: str, postgres_url: str):
    """Perform pre-flight checks before migration."""
    logger.info("Running pre-flight checks...")
    issues = []

    # Check DuckDB file exists
    if not Path(duckdb_path).exists():
        issues.append(f"DuckDB file not found: {duckdb_path}")
    else:
        logger.success(f"✓ DuckDB file found: {duckdb_path}")

    # Check PostgreSQL connection
    try:
        engine = create_engine(postgres_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.success("✓ PostgreSQL connection successful")
    except Exception as e:
        issues.append(f"Cannot connect to PostgreSQL: {e}")

    # Check pgvector extension
    try:
        engine = create_engine(postgres_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname='vector'")
            ).first()
            if result:
                logger.success("✓ pgvector extension is installed")
            else:
                logger.warning(
                    "⚠ pgvector extension not found, will attempt to install"
                )
    except Exception as e:
        issues.append(f"Error checking pgvector extension: {e}")

    if issues:
        logger.error("Pre-flight checks failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False

    logger.success("All pre-flight checks passed!")
    return True


def encode_postgres_url(postgres_url: str) -> str:
    """
    Properly URL-encode the password in PostgreSQL connection URL.
    Handles special characters like /, +, = in passwords.

    :param postgres_url: PostgreSQL connection URL
    :return: URL with properly encoded password
    """
    try:
        parsed = urlparse(postgres_url)

        if parsed.password:
            # URL-encode the password
            encoded_password = quote(parsed.password, safe="")

            # Reconstruct the URL with encoded password
            encoded_url = parsed._replace(
                netloc=f"{parsed.username}:{encoded_password}@{parsed.hostname}"
                f":{parsed.port}"
                if parsed.port
                else f"{parsed.username}:{encoded_password}@{parsed.hostname}"
            ).geturl()

            return encoded_url
    except Exception as e:
        logger.warning(f"Could not parse PostgreSQL URL, using as-is: {e}")

    return postgres_url


def inspect_duckdb_database(duck_conn):
    """Inspect DuckDB database structure."""
    logger.info("Inspecting DuckDB database...")

    tables = duck_conn.execute("SHOW TABLES").fetchall()
    table_names = [table[0] for table in tables]
    logger.info(f"Found tables: {table_names}")

    # Check for embedding table (support both old and new naming)
    embedding_table = None
    for table in ["embedding_duckdb", "embedding"]:
        if table in table_names:
            embedding_table = table
            break

    if not embedding_table:
        logger.error("No embedding table found in DuckDB")
        return None, None

    # Get table info
    count_result = duck_conn.execute(
        f"SELECT COUNT(*) FROM {embedding_table}"
    ).fetchone()
    total_records = count_result[0]

    # Get columns
    columns_result = duck_conn.execute(f"DESCRIBE {embedding_table}").fetchall()
    columns = [col[0] for col in columns_result]

    logger.info(f"Table: {embedding_table}")
    logger.info(f"Columns: {columns}")
    logger.info(f"Total records: {total_records}")

    return embedding_table, total_records


def migrate_embeddings(
    duckdb_path: str,
    postgres_url: str,
    batch_size: int = 1000,
    dry_run: bool = False,
    resume: bool = False,
    re_embed: bool = False,
    embedding_model: str = None,
):
    """
    Migrate embeddings from DuckDB to PostgreSQL.

    :param duckdb_path: Path to DuckDB database file
    :param postgres_url: PostgreSQL connection URL
    :param batch_size: Number of records to process in each batch
    :param dry_run: Preview migration without making changes
    :param resume: Resume from last position (skip existing records)
    :param re_embed: Re-embed documents if dimension mismatch detected
    :param embedding_model: Embedding model to use for re-embedding (default: from settings)
    """
    stats = MigrationStats()

    # URL-encode the PostgreSQL URL to handle special characters in password
    encoded_postgres_url = encode_postgres_url(postgres_url)

    # Pre-flight checks
    if not check_prerequisites(duckdb_path, encoded_postgres_url):
        sys.exit(1)

    # Connect to DuckDB
    logger.info(f"Connecting to DuckDB: {duckdb_path}")
    duck_conn = duckdb.connect(duckdb_path, read_only=True)

    # Inspect DuckDB
    embedding_table, total_records = inspect_duckdb_database(duck_conn)
    if not embedding_table:
        duck_conn.close()
        sys.exit(1)

    stats.total_records = total_records

    if total_records == 0:
        logger.info("No records to migrate. DuckDB database is empty.")
        duck_conn.close()
        return

    # Check embedding dimensions
    dim_match, duckdb_dim = check_embedding_dimension(duck_conn, embedding_table)

    if not dim_match and not re_embed:
        logger.error(f"Embedding dimension mismatch detected!")
        logger.error(f"  DuckDB dimension: {duckdb_dim}")
        logger.error(f"  PostgreSQL dimension: 1024")
        logger.error("")
        logger.error("Options to resolve this:")
        logger.error("  1. Re-embed documents: Add --re-embed flag")
        logger.error("  2. Cancel migration and investigate manually")
        logger.error("")
        logger.error("To re-embed with default model:")
        logger.error("  ./scripts/migrate.sh --re-embed")
        logger.error("")
        logger.error("To specify a different embedding model:")
        logger.error(
            "  python scripts/migrate_duckdb_to_postgres.py --re-embed --embedding-model mxbai-embed-large ..."
        )
        duck_conn.close()
        sys.exit(1)

    # Re-embed if needed
    if re_embed and not dim_match:
        logger.info("")
        logger.info("=" * 60)
        logger.info("RE-EMBEDDING MODE")
        logger.info("=" * 60)

        # Import DocEmbedder for re-embedding
        from libbydbot.brain.embed import DocEmbedder
        from libbydbot.settings import Settings

        # Get embedding model - use mxbai-embed-large for 1024 dimensions
        if not embedding_model:
            # Use mxbai-embed-large by default for re-embedding (1024 dimensions)
            embedding_model = "mxbai-embed-large"
            logger.info(
                "No embedding model specified, using mxbai-embed-large (1024 dimensions) for re-embedding"
            )

        logger.info(f"Using embedding model: {embedding_model}")

        # Initialize DocEmbedder for PostgreSQL (collection_name will be set per-document)
        doc_embedder = DocEmbedder(
            col_name="temp",  # Temporary, will be overridden per document
            dburl=encoded_postgres_url,
            embedding_model=embedding_model,
        )

        logger.info("Fetching documents from DuckDB for re-embedding...")
        documents_query = f"""
            SELECT DISTINCT doc_name, document, page_number, collection_name
            FROM {embedding_table}
            ORDER BY doc_name, page_number
        """
        documents = duck_conn.execute(documents_query).fetchall()

        logger.info(f"Found {len(documents)} unique document chunks to re-embed")

        # Get unique collection names for logging
        collections = set(doc[3] for doc in documents)
        logger.info(f"Collections to migrate: {collections}")

        if not dry_run:
            logger.info("Starting re-embedding process...")
            re_embedded = 0

            with tqdm(total=len(documents), desc="Re-embedding documents") as pbar:
                for doc_name, document, page_number, collection_name in documents:
                    try:
                        # Set the collection name for this document (preserves original collection)
                        doc_embedder.collection_name = collection_name

                        # Embed the document with new model
                        doc_embedder.embed_text(
                            doctext=document, docname=doc_name, page_number=page_number
                        )
                        re_embedded += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.error(
                            f"Error re-embedding {doc_name} page {page_number}: {e}"
                        )
                        stats.error_count += 1

            logger.success(
                f"Re-embedding complete! {re_embedded} documents re-embedded"
            )
            logger.info("")
            logger.info("Migration with re-embedding completed!")
            logger.info(f"Total documents re-embedded: {re_embedded}")
            logger.info(f"New embedding model: {embedding_model}")
            logger.info(f"New embedding dimension: 1024")

            duck_conn.close()
            return
        else:
            logger.info("DRY RUN: Would re-embed documents with new model")
            logger.info(f"Documents to re-embed: {len(documents)}")
            logger.info(f"New embedding model: {embedding_model}")
            logger.info(f"New embedding dimension: 1024")

    # Connect to PostgreSQL
    safe_url = (
        encoded_postgres_url.split("@")[1]
        if "@" in encoded_postgres_url
        else encoded_postgres_url
    )
    logger.info(f"Connecting to PostgreSQL: {safe_url}")
    pg_engine = create_engine(encoded_postgres_url)

    # Enable pgvector extension
    logger.info("Ensuring pgvector extension is enabled...")
    with pg_engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            logger.success("pgvector extension enabled")
        except Exception as e:
            logger.error(f"Failed to enable pgvector extension: {e}")
            logger.error("Make sure you're using the pgvector/pgvector Docker image")
            duck_conn.close()
            sys.exit(1)

    # Create tables if they don't exist
    if not dry_run:
        logger.info("Creating tables in PostgreSQL if they don't exist...")
        Base.metadata.create_all(pg_engine, checkfirst=True)
        logger.success("Tables ready")
    else:
        logger.info("DRY RUN: Skipping table creation")

    # Check existing records in PostgreSQL
    with Session(pg_engine) as session:
        try:
            existing_count = session.execute(
                text("SELECT COUNT(*) FROM embedding")
            ).scalar()
            if existing_count > 0:
                logger.info(f"Found {existing_count} existing records in PostgreSQL")
                if not resume:
                    response = input(
                        "Continue with migration? Existing records will be skipped. [Y/n]: "
                    )
                    if response.lower() == "n":
                        logger.info("Migration cancelled by user")
                        duck_conn.close()
                        sys.exit(0)
        except Exception as e:
            # Table doesn't exist yet, which is fine for migration
            logger.info(
                "No existing records in PostgreSQL (table will be created during migration)"
            )

    # Define columns to migrate
    columns = [
        "collection_name",
        "doc_name",
        "page_number",
        "doc_hash",
        "document",
        "embedding_model",
        "embedding",
    ]

    # Start migration
    logger.info(f"Starting migration of {total_records} records...")
    if dry_run:
        logger.warning("DRY RUN MODE: No changes will be made to PostgreSQL")

    stats.start()
    offset = 0
    processed = 0

    with Session(pg_engine) as pg_session:
        with tqdm(
            total=total_records,
            desc="Migrating embeddings",
            unit="records",
            postfix={"rate": "0 rec/s"},
        ) as pbar:
            while offset < total_records:
                batch_query = f"""
                    SELECT {", ".join(columns)}
                    FROM {embedding_table}
                    ORDER BY id
                    LIMIT {batch_size} OFFSET {offset}
                """

                batch = duck_conn.execute(batch_query).fetchall()

                if not batch:
                    break

                batch_start = time.time()

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

                    # Check for duplicates
                    if resume or dry_run:
                        try:
                            existing = pg_session.execute(
                                text("SELECT id FROM embedding WHERE doc_hash = :hash"),
                                {"hash": doc_hash},
                            ).first()

                            if existing:
                                stats.skipped_count += 1
                                processed += 1
                                pbar.update(1)
                                continue
                        except Exception:
                            # Table doesn't exist yet in dry-run mode,                            # which is fine
                            pass
                        except Exception:
                            # Table doesn't exist yet, which is expected for first migration
                            # or when running with --resume on an empty database
                            pass

                    if not dry_run:
                        # Create embedding record
                        embedding_record = Embedding(
                            collection_name=collection_name,
                            doc_name=doc_name,
                            page_number=page_number,
                            doc_hash=doc_hash,
                            document=document,
                            embedding_model=embedding_model or "embeddinggemma",
                            embedding=embedding,
                        )

                        try:
                            pg_session.add(embedding_record)
                            stats.migrated_count += 1
                        except Exception as e:
                            logger.error(f"Error adding record {doc_hash}: {e}")
                            stats.error_count += 1
                    else:
                        stats.migrated_count += 1

                    processed += 1
                    pbar.update(1)

                # Commit batch
                if not dry_run:
                    try:
                        pg_session.commit()
                        stats.batch_count += 1
                    except IntegrityError as e:
                        logger.warning(f"Batch commit error (duplicates): {e}")
                        pg_session.rollback()
                    except Exception as e:
                        logger.error(f"Error committing batch: {e}")
                        pg_session.rollback()
                        stats.error_count += len(batch)

                # Update progress bar with rate
                batch_time = time.time() - batch_start
                rate = len(batch) / batch_time if batch_time > 0 else 0
                pbar.set_postfix({"rate": f"{rate:.1f} rec/s"})

                offset += batch_size

    duck_conn.close()

    # Verify migration (if not dry run)
    if not dry_run:
        logger.info("Verifying migration...")
        with Session(pg_engine) as session:
            final_count = session.execute(
                text("SELECT COUNT(*) FROM embedding")
            ).scalar()
            logger.info(f"PostgreSQL now contains {final_count} records")

            if final_count < stats.migrated_count + existing_count:
                logger.warning(
                    "Record count mismatch! Some records may not have been migrated"
                )

    # Print summary
    print(stats.summary())

    if dry_run:
        logger.info("DRY RUN COMPLETE: No changes were made to PostgreSQL")
        logger.info(f"To perform actual migration, run without --dry-run flag")

    if stats.error_count > 0:
        logger.warning(f"Migration completed with {stats.error_count} errors")
        sys.exit(1)

    logger.success("Migration completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate embeddings from DuckDB to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview migration
  %(prog)s --dry-run --duckdb-path ./embeddings.duckdb --postgres-url postgresql://libby:pass@localhost:5432/libby

  # Run migration
  %(prog)s --duckdb-path ./embeddings.duckdb --postgres-url postgresql://libby:pass@localhost:5432/libby

  # Resume interrupted migration
  %(prog)s --resume --duckdb-path ./embeddings.duckdb --postgres-url postgresql://libby:pass@localhost:5432/libby

  # Re-embed documents when dimension mismatch detected
  %(prog)s --re-embed --duckdb-path ./embeddings.duckdb --postgres-url postgresql://libby:pass@localhost:5432/libby

  # Re-embed with specific model
  %(prog)s --re-embed --embedding-model mxbai-embed-large --duckdb-path ./embeddings.duckdb --postgres-url postgresql://libby:pass@localhost:5432/libby

  # Fast migration with larger batches
  %(prog)s --duckdb-path ./embeddings.duckdb --postgres-url postgresql://libby:pass@localhost:5432/libby --batch-size 5000
        """,
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes to PostgreSQL",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume migration, skipping already migrated records",
    )
    parser.add_argument(
        "--re-embed",
        action="store_true",
        help="Re-embed documents if dimension mismatch detected (uses default embedding model)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model to use for re-embedding (default: from settings, usually mxbai-embed-large)",
    )

    args = parser.parse_args()

    # Print header
    print(f"\n{'=' * 60}")
    print(f"DuckDB to PostgreSQL Migration Tool")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    try:
        migrate_embeddings(
            args.duckdb_path,
            args.postgres_url,
            args.batch_size,
            args.dry_run,
            args.resume,
            args.re_embed,
            args.embedding_model,
        )
    except KeyboardInterrupt:
        logger.warning("\nMigration interrupted by user")
        logger.info("You can resume migration with --resume flag")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
