#!/usr/bin/env python3
"""
Check if all vectors in the DuckDB embedding database have the same length.
"""

import sys
import duckdb


def check_duckdb_vectors(db_path: str):
    """Check if all vectors have the same length in DuckDB."""
    print(f"Checking database: {db_path}")
    print("=" * 60)

    try:
        conn = duckdb.connect(db_path, read_only=True)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # Get list of tables
    tables = conn.sql(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    print(f"\nTables found: {[t[0] for t in tables]}")

    # Check for embedding tables
    embedding_tables = [
        t[0] for t in tables if "embedding" in t[0].lower() or "vec" in t[0].lower()
    ]

    if not embedding_tables:
        print("\nNo embedding tables found")
        conn.close()
        return

    for table_name in embedding_tables:
        print(f"\n{'=' * 60}")
        print(f"Checking table: {table_name}")
        print("=" * 60)

        # Get total count
        try:
            count_result = conn.sql(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            total_count = count_result[0]
            print(f"Total embeddings: {total_count}")

            if total_count == 0:
                print("No embeddings in this table")
                continue
        except Exception as e:
            print(f"Error counting rows: {e}")
            continue

        # Get column info
        columns_info = conn.sql(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """).fetchall()

        print(f"\nColumns in {table_name}:")
        for col_name, col_type in columns_info:
            print(f"  - {col_name}: {col_type}")

        # Find the embedding column (must have FLOAT[] type)
        embedding_col = None
        for col_name, col_type in columns_info:
            if "FLOAT[" in col_type:
                embedding_col = col_name
                break

        if not embedding_col:
            print("Could not identify embedding column")
            continue

        print(f"\nUsing embedding column: {embedding_col}")

        # Check dimensions using array_length
        query = f"""
            SELECT 
                array_length({embedding_col}) as dimension,
                COUNT(*) as count
            FROM {table_name}
            GROUP BY array_length({embedding_col})
            ORDER BY dimension;
        """

        try:
            results = conn.sql(query).fetchall()
        except Exception as e:
            print(f"Error querying dimensions: {e}")
            continue

        if not results:
            print("No embedding data found")
            continue

        print(f"\nVector dimensions found:")
        print("-" * 50)

        dimensions = []
        for dim, count in results:
            print(f"Dimension: {dim:4d} | Count: {count:5d}")
            dimensions.append(dim)

        print("-" * 50)

        if len(set(dimensions)) == 1:
            print(f"\n✓ All vectors have the same length: {dimensions[0]}")
        else:
            print(f"\n✗ WARNING: Vectors have different lengths!")
            print(f"  Unique dimensions: {sorted(set(dimensions))}")
            print(f"  Min dimension: {min(dimensions)}")
            print(f"  Max dimension: {max(dimensions)}")

            # Show samples of mismatched dimensions
            most_common_dim = max(results, key=lambda x: x[1])[0]

            print(
                f"\n  Sample of vectors with non-standard dimensions (not {most_common_dim}):"
            )
            sample_query = f"""
                SELECT doc_name, page_number, array_length({embedding_col}) as dim
                FROM {table_name}
                WHERE array_length({embedding_col}) != {most_common_dim}
                LIMIT 10;
            """

            try:
                samples = conn.sql(sample_query).fetchall()
                if samples:
                    for doc_name, page_num, dim in samples:
                        print(f"    - {doc_name} (page {page_num}): dim={dim}")
                else:
                    print("    No samples found")
            except Exception as e:
                print(f"    Error getting samples: {e}")

    conn.close()


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "embeddings.duckdb"
    check_duckdb_vectors(db_path)
