# Migration Guide: DuckDB to PostgreSQL

This guide will help you migrate your existing Libby D. Bot embeddings from DuckDB to PostgreSQL.

## Prerequisites

1. PostgreSQL with pgvector extension running (included in docker-compose.yml)
2. Python environment with Libby D. Bot installed
3. Access to your existing DuckDB database file

## Step 1: Set Up PostgreSQL

### Option A: Using Docker Compose (Recommended)

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set a secure password
nano .env
# Set POSTGRES_PASSWORD=your_secure_password_here

# Start PostgreSQL service
docker compose up -d postgres

# Wait for PostgreSQL to be ready
docker compose logs -f postgres
# Look for: "database system is ready to accept connections"
```

### Option B: Using Existing PostgreSQL

If you have an existing PostgreSQL instance with pgvector:

```bash
# Ensure pgvector extension is installed
psql -U your_user -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Step 2: Locate Your DuckDB File

Find your existing DuckDB database file. Common locations:

- Default location: `./embeddings.duckdb`
- Docker volume: Check your docker-compose.yml or run:
  ```bash
  docker volume inspect libby_libby-data
  ```

## Step 3: Run the Migration

### If using Docker Compose PostgreSQL:

```bash
# Get the password from your .env file
POSTGRES_PASSWORD=$(grep POSTGRES_PASSWORD .env | cut -d'=' -f2)

# Run migration
uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:${POSTGRES_PASSWORD}@localhost:5432/libby" \
  --batch-size 1000
```

### If using external PostgreSQL:

```bash
uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path /path/to/your/embeddings.duckdb \
  --postgres-url "postgresql://your_user:your_password@your_host:5432/your_database" \
  --batch-size 1000
```

## Step 4: Verify Migration

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U libby -d libby

# Count records
SELECT COUNT(*) FROM embedding;

# Check collections
SELECT collection_name, COUNT(*) as doc_count 
FROM embedding 
GROUP BY collection_name;

# Exit psql
\q
```

## Step 5: Update Configuration

Your `.env` file should already have PostgreSQL configuration:

```env
POSTGRES_DB=libby
POSTGRES_USER=libby
POSTGRES_PASSWORD=your_secure_password
EMBED_DB=postgresql://libby:your_secure_password@postgres:5432/libby
```

## Step 6: Start Using PostgreSQL

```bash
# Start all services
docker compose up -d

# Check API health
curl http://localhost:8001/api/health
```

## Troubleshooting

### Error: "pgvector extension not found"

**Solution**: Make sure you're using the pgvector/pgvector Docker image or have pgvector installed in your PostgreSQL instance.

```bash
# In psql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Error: "Connection refused"

**Solution**: Ensure PostgreSQL is running and accessible.

```bash
# Check if PostgreSQL is running
docker compose ps postgres

# Check logs
docker compose logs postgres
```

### Error: "Database 'libby' does not exist"

**Solution**: The database will be created automatically by the PostgreSQL container. If using an external PostgreSQL:

```bash
psql -U postgres -c "CREATE DATABASE libby;"
```

### Migration is slow

**Solution**: Increase batch size or check network connectivity:

```bash
# Increase batch size to 5000
uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path ./embeddings.duckdb \
  --postgres-url "postgresql://libby:${POSTGRES_PASSWORD}@localhost:5432/libby" \
  --batch-size 5000
```

## Rollback Plan

If you need to rollback to DuckDB:

1. Your original DuckDB file is not modified by the migration
2. Simply update your `.env` to use DuckDB:

```env
EMBED_DB=duckdb:///data/embeddings.duckdb
```

3. Restart services:

```bash
docker compose restart libby-api
```

## Performance Tips

After migration, consider these optimizations:

1. **Create indexes** (automatic with pgvector):
   ```sql
   -- Index is created automatically, but you can verify
   \di embedding*
   ```

2. **Vacuum and analyze**:
   ```sql
   VACUUM ANALYZE embedding;
   ```

3. **Monitor performance**:
   ```sql
   -- Check table size
   SELECT pg_size_pretty(pg_total_relation_size('embedding'));
   
   -- Check index usage
   SELECT * FROM pg_stat_user_indexes WHERE relname = 'embedding';
   ```

## Need Help?

If you encounter issues:

1. Check the logs: `docker compose logs -f`
2. Verify your connection string
3. Ensure pgvector extension is installed
4. Open an issue at: https://github.com/your-repo/libby/issues
