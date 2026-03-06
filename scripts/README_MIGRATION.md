# DuckDB to PostgreSQL Migration Script

This script provides a robust, feature-rich migration tool for transferring embeddings from DuckDB to PostgreSQL with pgvector.

## Features

- ✅ **Pre-flight checks**: Validates both databases before migration
- ✅ **Dry-run mode**: Preview migration without making changes
- ✅ **Progress tracking**: Real-time progress with time estimation
- ✅ **Automatic duplicate detection**: Skips existing records
- ✅ **Resume capability**: Can resume interrupted migrations
- ✅ **Data integrity verification**: Validates migration results
- ✅ **Detailed reporting**: Comprehensive migration summary
- ✅ **Error handling**: Graceful error recovery with rollback

## Prerequisites

1. **PostgreSQL with pgvector** running (included in docker-compose.yml)
2. **Python environment** with Libby D. Bot installed
3. **Access** to your existing DuckDB database file

## Quick Start

### 1. Start PostgreSQL

```bash
# Start PostgreSQL service
docker compose up -d postgres

# Wait for it to be ready
docker compose logs -f postgres
```

### 2. Run Migration

```bash
# Get password from .env
export POSTGRES_PASSWORD=$(grep POSTGRES_PASSWORD .env | cut -d'=' -f2)

# Preview migration (dry-run)
uv run python scripts/migrate_duckdb_to_postgres.py \
  --dry-run \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:${POSTGRES_PASSWORD}@localhost:5432/libby"

# Run actual migration
uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:${POSTGRES_PASSWORD}@localhost:5432/libby"
```

## Usage Options

### Basic Migration

```bash
uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path /path/to/embeddings.duckdb \
  --postgres-url "postgresql://user:password@host:5432/database"
```

### Dry-Run (Preview)

Preview migration without making any changes:

```bash
uv run python scripts/migrate_duckdb_to_postgres.py \
  --dry-run \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:password@localhost:5432/libby"
```

### Resume Interrupted Migration

If migration was interrupted, resume from where it left off:

```bash
uv run python scripts/migrate_duckdb_to_postgres.py \
  --resume \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:password@localhost:5432/libby"
```

### Fast Migration (Larger Batches)

For faster migration on large datasets:

```bash
uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:password@localhost:5432/libby" \
  --batch-size 5000
```

## Command Line Options

| Option | Required | Description |
|--------|----------|-------------|
| `--duckdb-path` | Yes | Path to DuckDB database file |
| `--postgres-url` | Yes | PostgreSQL connection URL |
| `--batch-size` | No | Records per batch (default: 1000) |
| `--dry-run` | No | Preview without making changes |
| `--resume` | No | Resume from last position |
| `--help` | No | Show help message |

## Migration Process

The script performs these steps:

1. **Pre-flight checks**
   - Verifies DuckDB file exists
   - Tests PostgreSQL connection
   - Checks pgvector extension

2. **Database inspection**
   - Scans DuckDB table structure
   - Counts total records
   - Identifies existing records in PostgreSQL

3. **Migration**
   - Processes records in batches
   - Detects and skips duplicates
   - Shows real-time progress
   - Handles errors gracefully

4. **Verification**
   - Validates record counts
   - Checks data integrity
   - Generates summary report

## Example Output

```
============================================================
DuckDB to PostgreSQL Migration Tool
Started at: 2026-03-06 15:30:00
============================================================

Running pre-flight checks...
✓ DuckDB file found: ./embeddings.duckdb
✓ PostgreSQL connection successful
✓ pgvector extension is installed
All pre-flight checks passed!

Inspecting DuckDB database...
Found tables: ['embedding_duckdb']
Table: embedding_duckdb
Columns: ['id', 'collection_name', 'doc_name', 'page_number', 'doc_hash', 'document', 'embedding_model', 'embedding']
Total records: 15432

Starting migration of 15432 records...
Migrating embeddings: 100%|██████████| 15432/15432 [00:45<00:00, 342.1 rec/s]

============================================================
Migration Summary
============================================================
Total records processed:  15432
Successfully migrated:    15200
Skipped (duplicates):     232
Errors:                   0
Batches processed:        16
Total time:               45.12 seconds
Average rate:             341.95 records/sec
============================================================

Migration completed successfully!
```

## Troubleshooting

### Error: "pgvector extension not found"

**Solution**: Ensure you're using the pgvector/pgvector Docker image:

```bash
docker compose down -v
docker compose up -d postgres
```

### Error: "Connection refused"

**Solution**: Check if PostgreSQL is running:

```bash
docker compose ps postgres
docker compose logs postgres
```

### Error: "DuckDB file not found"

**Solution**: Verify the path to your DuckDB file:

```bash
# Find DuckDB files
find . -name "*.duckdb" -type f

# Check Docker volumes
docker volume inspect libby_libby-data
```

### Migration is slow

**Solutions**:
1. Increase batch size: `--batch-size 5000`
2. Check network connectivity
3. Ensure PostgreSQL has sufficient resources

### Some records failed to migrate

**Solution**: Check error logs and re-run with `--resume`:

```bash
uv run python scripts/migrate_duckdb_to_postgres.py \
  --resume \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:password@localhost:5432/libby"
```

## Best Practices

1. **Always backup before migration**
   ```bash
   cp embeddings.duckdb embeddings.duckdb.backup
   ```

2. **Test with dry-run first**
   ```bash
   uv run python scripts/migrate_duckdb_to_postgres.py --dry-run ...
   ```

3. **Verify after migration**
   ```bash
   docker compose exec postgres psql -U libby -d libby -c "SELECT COUNT(*) FROM embedding;"
   ```

4. **Monitor resources during migration**
   ```bash
   docker stats
   ```

## Performance Tips

- **Batch size**: Use larger batches (5000+) for faster migration
- **Network**: Run migration close to PostgreSQL (same machine/network)
- **Resources**: Ensure PostgreSQL has enough memory and CPU
- **Indexes**: Indexes are created automatically by pgvector

## Getting Help

If you encounter issues:

1. Check logs: `docker compose logs -f`
2. Review this README
3. Consult `docs/MIGRATION.md`
4. Open an issue on GitHub

## See Also

- [Migration Guide](../docs/MIGRATION.md) - Complete migration documentation
- [PostgreSQL Setup](../README.md#database-configuration) - Database configuration details
