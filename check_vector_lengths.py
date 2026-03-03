#!/usr/bin/env python3
"""
Check if all vectors in the embedding database have the same length.
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

load_dotenv()


def check_vector_lengths():
    """Check if all vectors have the same length."""
    dburl = os.getenv("PGURL")

    if not dburl:
        print("Error: PGURL environment variable not set")
        sys.exit(1)

    print(f"Connecting to database...")
    engine = create_engine(dburl)

    with Session(engine) as session:
        # Check if embedding table exists
        result = session.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename='embedding';")
        ).fetchone()

        if not result:
            print("No 'embedding' table found in the database")
            return

        print("Found 'embedding' table")

        # Get total count of embeddings
        count_result = session.execute(text("SELECT COUNT(*) FROM embedding")).scalar()
        print(f"Total embeddings: {count_result}")

        if count_result == 0:
            print("No embeddings in the database")
            return

        # Check vector dimensions using pgvector's vector_dims function
        query = text("""
            SELECT 
                vector_dims(embedding) as dimension,
                COUNT(*) as count,
                embedding_model
            FROM embedding
            GROUP BY vector_dims(embedding), embedding_model
            ORDER BY dimension;
        """)

        results = session.execute(query).fetchall()

        if len(results) == 0:
            print("No embeddings found")
            return

        print(f"\nVector dimensions found:")
        print("-" * 60)

        dimensions = []
        for dim, count, model in results:
            print(
                f"Dimension: {dim:4d} | Count: {count:5d} | Model: {model or 'unknown'}"
            )
            dimensions.append(dim)

        print("-" * 60)

        if len(set(dimensions)) == 1:
            print(f"\n✓ All vectors have the same length: {dimensions[0]}")
        else:
            print(f"\n✗ WARNING: Vectors have different lengths!")
            print(f"  Unique dimensions: {sorted(set(dimensions))}")
            print(f"  Min dimension: {min(dimensions)}")
            print(f"  Max dimension: {max(dimensions)}")

            # Show some examples of mismatched dimensions
            print("\n  Sample of vectors with non-standard dimensions:")
            sample_query = text("""
                SELECT doc_name, page_number, vector_dims(embedding) as dim, embedding_model
                FROM embedding
                WHERE vector_dims(embedding) != :most_common_dim
                LIMIT 10;
            """)

            # Find most common dimension
            most_common_dim = max(results, key=lambda x: x[1])[0]

            samples = session.execute(
                sample_query, {"most_common_dim": most_common_dim}
            ).fetchall()

            if samples:
                for doc_name, page_num, dim, model in samples:
                    print(
                        f"    - {doc_name} (page {page_num}): dim={dim}, model={model or 'unknown'}"
                    )


if __name__ == "__main__":
    check_vector_lengths()
