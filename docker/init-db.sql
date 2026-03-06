-- Enable pgvector extension for vector similarity search
-- This script runs automatically when PostgreSQL container initializes
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is loaded successfully
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'pgvector extension could not be enabled. Ensure you are using pgvector/pgvector Docker image';
    END IF;
    RAISE NOTICE 'pgvector extension successfully enabled in database';
END $$;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
END $$;
