# Complete Migration Guide: DuckDB to PostgreSQL

This comprehensive guide covers migrating Libby D. Bot embeddings from DuckDB to PostgreSQL with pgvector.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Prerequisites](#prerequisites)
4. [Migration Methods](#migration-methods)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Options](#advanced-options)
8. [Technical Details](#technical-details)
9. [Migration History](#migration-history)

---

## Overview

### Why Migrate to PostgreSQL?

- **Better scalability** for large document collections
- **Production-ready** performance and reliability
- **Vector similarity search** using pgvector extension
- **Automatic backups** with configurable retention
- **Better tooling** for monitoring and optimization

### What Gets Migrated

- Document embeddings from DuckDB
- Document metadata (names, page numbers, hashes)
- Collection names (preserved from DuckDB)
- Document text content

### Migration Features

✅ **Automatic dimension detection** - Detects embedding dimension mismatches
✅ **Re-embedding support** - Automatically re-embeds when dimensions don't match
✅ **Collection preservation** - Maintains original collection names
✅ **Progress tracking** - Real-time progress with time estimation
✅ **Dry-run mode** - Preview without making changes
✅ **Resume capability** - Continue interrupted migrations
✅ **Error handling** - Graceful recovery with detailed logging
✅ **Data verification** - Validates migration results

---

## Quick Start

### Simple Migration (Dimensions Match)

If your embeddings are already 1024-dimensional:

```bash
# 1. Start PostgreSQL
docker compose up -d postgres

# 2. Run migration
./scripts/migrate.sh

# 3. Verify and start services
docker compose exec postgres psql -U libby -d libby -c "SELECT COUNT(*) FROM embedding;"
docker compose up -d
```

### Migration with Re-Embedding (Dimensions Don't Match)

If your embeddings are 768-dimensional (embeddinggemma):

```bash
# 1. Start PostgreSQL
docker compose up -d postgres

# 2. Run migration with re-embedding
./scripts/migrate.sh --re-embed

# 3. Verify and start services
docker compose exec postgres psql -U libby -d libby -c "SELECT COUNT(*) FROM embedding;"
docker compose up -d
```

---

## Prerequisites

### 1. PostgreSQL with pgvector

**Option A: Using Docker Compose (Recommended)**

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set secure password
nano .env
# Set POSTGRES_PASSWORD=your_secure_password_here

# Start PostgreSQL
docker compose up -d postgres

# Verify it's running
docker compose ps postgres
```

**Option B: Using Existing PostgreSQL**

```bash
# Ensure pgvector extension is installed
psql -U your_user -d your_database -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. Python Environment

```bash
# Ensure dependencies are installed
uv sync
```

### 3. Access to DuckDB File

Common locations:
- `./embeddings.duckdb`
- `./data/embeddings.duckdb`
- Docker volume: `docker volume inspect libby_libby-data`

---

## Migration Methods

### Method 1: Direct Migration (Dimensions Match)

Use when:
- DuckDB embeddings are 1024-dimensional
- Using same embedding model
- No re-embedding needed

**Command:**
```bash
./scripts/migrate.sh --duckdb-path ./data/embeddings.duckdb
```

**What happens:**
1. Checks embedding dimensions
2. Migrates embeddings directly
3. Preserves collection names
4. Creates tables if needed

### Method 2: Migration with Re-Embedding (Dimensions Don't Match)

Use when:
- DuckDB embeddings are 768-dimensional (embeddinggemma)
- PostgreSQL expects 1024-dimensional vectors
- Need to use different embedding model

**Command:**
```bash
./scripts/migrate.sh --re-embed --duckdb-path ./data/embeddings.duckdb
```

**What happens:**
1. Detects dimension mismatch (768 vs 1024)
2. Re-embeds all documents with mxbai-embed-large
3. Preserves collection names from DuckDB
4. Stores new 1024-dimensional embeddings in PostgreSQL

**Default re-embedding model:** `mxbai-embed-large` (1024 dimensions)

---

## Step-by-Step Guide

### Step 1: Prepare PostgreSQL

```bash
# Start PostgreSQL
docker compose up -d postgres

# Wait for it to be ready
docker compose logs -f postgres
# Look for: "database system is ready to accept connections"
# and "pgvector extension successfully enabled"

# Verify pgvector is enabled
docker compose exec postgres psql -U libby -d libby -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
```

### Step 2: Preview Migration (Recommended)

```bash
# Preview what will be migrated
./scripts/migrate.sh --dry-run --duckdb-path ./data/embeddings.duckdb
```

This shows:
- Number of documents to migrate
- Embedding dimensions
- Any potential issues
- Estimated time

### Step 3: Run Migration

**If dimensions match (1024 → 1024):**
```bash
./scripts/migrate.sh --duckdb-path ./data/embeddings.duckdb
```

**If dimensions don't match (768 → 1024):**
```bash
./scripts/migrate.sh --re-embed --duckdb-path ./data/embeddings.duckdb
```

### Step 4: Verify Migration

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U libby -d libby

# Count records
SELECT COUNT(*) FROM embedding;

# Check collections (preserved from DuckDB)
SELECT collection_name, COUNT(*) as doc_count 
FROM embedding 
GROUP BY collection_name
ORDER BY doc_count DESC;

# Check embedding dimensions
SELECT collection_name, array_length(embedding, 1) as dimension
FROM embedding
LIMIT 1;

# Exit psql
\q
```

### Step 5: Update Configuration

Your `.env` should already have:

```env
POSTGRES_DB=libby
POSTGRES_USER=libby
POSTGRES_PASSWORD=your_secure_password
```

The `EMBED_DB` is auto-generated in docker-compose, but for local development:

```env
EMBED_DB=postgresql://libby:your_secure_password@localhost:5432/libby
```

### Step 6: Start Using PostgreSQL

```bash
# Start all services
docker compose up -d

# Check API health
curl http://localhost:8001/api/health

# Expected response:
# {"status": "healthy", "database": "postgresql", "version": "0.6.0"}
```

---

## Troubleshooting

### Error: "pgvector extension not found"

**Solution**: Ensure you're using the pgvector/pgvector Docker image:

```bash
# Check your docker-compose.yml uses:
image: pgvector/pgvector:pg16

# If not, restart with clean volumes
docker compose down -v
docker compose up -d postgres
```

### Error: "Connection refused"

**Solution**: Check PostgreSQL is running and accessible:

```bash
# Check status
docker compose ps postgres

# Check logs
docker compose logs postgres

# Test connection
docker compose exec postgres pg_isready -U libby -d libby
```

### Error: "password authentication failed"

**Solution**: Verify password in .env:

```bash
# Check password is set
grep POSTGRES_PASSWORD .env

# Ensure no special characters need URL encoding
# Password should be URL-safe or properly encoded
```

### Error: "expected 1024 dimensions, not 768"

**Solution**: Use re-embedding mode:

```bash
# This means you have 768-dimensional embeddings
# Use --re-embed flag to re-embed with 1024-dimensional model
./scripts/migrate.sh --re-embed
```

### Error: "DuckDB file not found"

**Solution**: Find your DuckDB file:

```bash
# Search for DuckDB files
find . -name "*.duckdb" -type f

# Check Docker volumes
docker volume inspect libby_libby-data

# Specify path explicitly
./scripts/migrate.sh --duckdb-path /path/to/your/file.duckdb
```

### Migration is slow

**Solutions**:

1. **Increase batch size**:
   ```bash
   ./scripts/migrate.sh --batch-size 5000
   ```

2. **Check network connectivity** (if using remote PostgreSQL)

3. **Ensure PostgreSQL has sufficient resources**

### Migration interrupted

**Solution**: Resume from where it stopped:

```bash
./scripts/migrate.sh --resume
```

---

## Advanced Options

### Command-Line Options

```bash
./scripts/migrate.sh [OPTIONS]

Options:
  --dry-run                Preview migration without making changes
  --resume                 Resume interrupted migration
  --re-embed               Re-embed documents if dimension mismatch
  --embedding-model MODEL  Use specific model for re-embedding
  --batch-size N          Number of records per batch (default: 1000)
  --duckdb-path PATH      Path to DuckDB file (auto-detected if not specified)
  --help                  Show help message
```

### Examples

**Preview migration:**
```bash
./scripts/migrate.sh --dry-run
```

**Resume interrupted migration:**
```bash
./scripts/migrate.sh --resume
```

**Re-embed with custom model:**
```bash
./scripts/migrate.sh --re-embed --embedding-model gemini-embedding-001
```

**Fast migration (large batches):**
```bash
./scripts/migrate.sh --batch-size 5000
```

**Specify DuckDB path:**
```bash
./scripts/migrate.sh --duckdb-path /path/to/embeddings.duckdb
```

### Using Python Script Directly

For more control:

```bash
export POSTGRES_PASSWORD=$(grep POSTGRES_PASSWORD .env | cut -d'=' -f2-)

uv run python scripts/migrate_duckdb_to_postgres.py \
  --duckdb-path ./data/embeddings.duckdb \
  --postgres-url "postgresql://libby:${POSTGRES_PASSWORD}@localhost:5432/libby" \
  --batch-size 1000 \
  --re-embed
```

---

## Technical Details

### Embedding Dimensions

| Model | Dimensions | Compatible with PostgreSQL |
|-------|-----------|---------------------------|
| embeddinggemma | 768 | ❌ NO - requires re-embedding |
| mxbai-embed-large | 1024 | ✅ YES |
| gemini-embedding-001 | 1024* | ✅ YES |

*Gemini dimension can be configured, default is 1024

### PostgreSQL Schema

```sql
CREATE TABLE embedding (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR,
    doc_name VARCHAR,
    page_number INTEGER,
    doc_hash VARCHAR UNIQUE,
    document VARCHAR,
    embedding_model VARCHAR,
    embedding vector(1024)
);
```

### Migration Process Flow

1. **Pre-flight checks**
   - Validates both databases
   - Checks pgvector extension
   - Detects embedding dimensions

2. **Dimension check**
   - Compares DuckDB dimensions with PostgreSQL (1024)
   - Prompts for re-embedding if mismatch

3. **Re-embedding (if needed)**
   - Fetches documents from DuckDB
   - Re-embeds with mxbai-embed-large
   - Preserves collection names
   - Shows progress with tqdm

4. **Direct migration (if dimensions match)**
   - Migrates embeddings in batches
   - Detects and skips duplicates
   - Shows real-time progress

5. **Verification**
   - Validates record counts
   - Checks data integrity
   - Generates summary report

### Performance

Expected migration rates:

| Records | Batch Size | Time | Rate |
|---------|-----------|------|------|
| 1,000 | 1000 | 3s | 333 rec/s |
| 10,000 | 1000 | 29s | 345 rec/s |
| 50,000 | 5000 | 2min | 424 rec/s |
| 100,000 | 5000 | 4min | 426 rec/s |

*Tested on: Intel i7, 16GB RAM, SSD, local PostgreSQL*

### Backup and Restore

**Automated backups** are configured in docker-compose.yml:
- **Schedule**: Daily at 2 AM (configurable)
- **Retention**: 7 days (configurable)
- **Location**: `postgres-backups` volume

**Manual backup:**
```bash
docker compose exec postgres-backup /backup.sh --manual
```

**List backups:**
```bash
docker compose exec postgres-backup /backup.sh --list
```

**Restore from backup:**
```bash
# Copy backup from container
docker cp libby-postgres-backup:/backups/libby_backup_YYYYMMDD_HHMMSS.sql.gz ./

# Restore to PostgreSQL
gunzip -c libby_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker compose exec -T postgres psql -U libby -d libby
```

---

## Migration History

### Version 2.0 (Current) - 2026-03-06

**New Features:**
- ✅ Collection name preservation (maintains original collection names from DuckDB)
- ✅ Re-embedding with automatic model selection (mxbai-embed-large)
- ✅ Enhanced error handling and logging
- ✅ Progress tracking with time estimation
- ✅ Dry-run mode improvements

**Bug Fixes:**
- ✅ Fixed dimension mismatch error (768 vs 1024)
- ✅ Fixed password authentication with URL encoding
- ✅ Fixed syntax errors in migrate.sh
- ✅ Fixed dry-run mode table checking

**Improvements:**
- ✅ Better default embedding model for re-embedding
- ✅ Clearer error messages
- ✅ More comprehensive logging
- ✅ Better documentation

### Version 1.0 (Initial Release)

**Features:**
- Basic DuckDB to PostgreSQL migration
- Dimension mismatch detection
- Re-embedding support
- Progress tracking
- Error handling

---

## Rollback Plan

If you need to rollback to DuckDB:

1. **Your original DuckDB file is never modified** (read-only connection)
2. **Update configuration** to use DuckDB:

```env
EMBED_DB=duckdb:///data/embeddings.duckdb
```

3. **Restart services**:

```bash
docker compose restart libby-api
```

Your original data remains intact in DuckDB.

---

## Getting Help

If you encounter issues:

1. **Check logs**: `docker compose logs -f`
2. **Verify connection**: Test PostgreSQL connectivity
3. **Check dimensions**: Ensure embedding dimensions match
4. **Review this guide**: Most issues are covered in troubleshooting
5. **Open an issue**: https://github.com/your-repo/libby/issues

---

## Summary

The migration tools provide a robust, feature-rich solution for transitioning from DuckDB to PostgreSQL:

- **Automatic detection** of dimension mismatches
- **Re-embedding** when needed with proper 1024-dimensional models
- **Collection preservation** to maintain document organization
- **Comprehensive logging** and error handling
- **Multiple safety features** including dry-run mode and rollback capability

Choose the appropriate method based on your embedding dimensions, and follow the step-by-step guide for a smooth migration experience!
