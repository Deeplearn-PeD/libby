# Libby D. Bot Documentation

## Getting Started

- **[Main README](../README.md)** - Start here for installation and basic usage
- **[AGENTS.md](../AGENTS.md)** - Development guidelines and code style

## Database & Migration

- **[Complete Migration Guide](COMPLETE_MIGRATION_GUIDE.md)** - Comprehensive DuckDB to PostgreSQL migration guide
  - Quick start instructions
  - Step-by-step migration process
  - Troubleshooting common issues
  - Advanced options and performance tuning

### Migration Documentation (Historical)

These documents contain historical information about migration development:

- **[MIGRATION.md](MIGRATION.md)** - Original migration guide
- **[MIGRATION_TOOLS.md](MIGRATION_TOOLS.md)** - Migration tools overview
- **[MIGRATION_FIX.md](MIGRATION_FIX.md)** - Password and syntax fixes
- **[RE_EMBED_FEATURE.md](RE_EMBED_FEATURE.md)** - Re-embedding feature details
- **[RE_EMBED_BUG_FIX.md](RE_EMBED_BUG_FIX.md)** - Dimension mismatch fix
- **[POSTGRESQL_MIGRATION_SUMMARY.md](POSTGRESQL_MIGRATION_SUMMARY.md)** - PostgreSQL migration summary

> **Note**: For current migration instructions, use the **[Complete Migration Guide](COMPLETE_MIGRATION_GUIDE.md)** which consolidates all the above information.

## Scripts

Migration scripts are located in `scripts/`:

- **`migrate.sh`** - User-friendly migration helper (recommended)
- **`migrate_duckdb_to_postgres.py`** - Python migration script (advanced)
- **`backup-postgres.sh`** - PostgreSQL backup automation

## Quick Links

### For Users

1. [Installation](../README.md#installation)
2. [Quick Start](../README.md#quick-start)
3. [API Documentation](../README.md#api-documentation)
4. [Migration Guide](COMPLETE_MIGRATION_GUIDE.md)

### For Developers

1. [Development Setup](../AGENTS.md)
2. [Code Style Guidelines](../AGENTS.md#code-style-guidelines)
3. [Testing](../AGENTS.md#running-tests)
4. [Release Process](../AGENTS.md#release-process)

## Need Help?

- Check the [Troubleshooting](COMPLETE_MIGRATION_GUIDE.md#troubleshooting) section
- Review [API Documentation](../README.md#api-documentation)
- Open an issue on GitHub
