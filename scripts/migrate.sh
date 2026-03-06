#!/bin/bash
#
# Quick migration helper script
# Simplifies the DuckDB to PostgreSQL migration process
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓ ${NC}$1"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${NC}$1"
}

print_error() {
    echo -e "${RED}✗ ${NC}$1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "       DuckDB to PostgreSQL Migration Helper"
    echo "============================================================"
    echo -e "${NC}"
}

# Check if .env exists
check_env() {
    if [ ! -f .env ]; then
        print_error ".env file not found"
        print_info "Creating .env from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env and set POSTGRES_PASSWORD, then run this script again"
        exit 1
    fi
    
    # Check if POSTGRES_PASSWORD is set
    if grep -q "POSTGRES_PASSWORD=CHANGE_ME" .env; then
        print_error "POSTGRES_PASSWORD not set in .env"
        print_info "Please edit .env and set a secure POSTGRES_PASSWORD"
        exit 1
    fi
    
    print_success ".env file found and configured"
}

# Check if PostgreSQL is running
check_postgres() {
    print_info "Checking PostgreSQL status..."
    
    if ! docker compose ps postgres | grep -q "Up"; then
        print_warning "PostgreSQL is not running"
        print_info "Starting PostgreSQL..."
        docker compose up -d postgres
        
        print_info "Waiting for PostgreSQL to be ready..."
        sleep 5
        
        for i in {1..30}; do
            if docker compose exec -T postgres pg_isready -U libby -d libby > /dev/null 2>&1; then
                print_success "PostgreSQL is ready"
                return 0
            fi
            sleep 1
        done
        
        print_error "PostgreSQL failed to start"
        exit 1
    else
        print_success "PostgreSQL is running"
    fi
}

# Find DuckDB file
find_duckdb_file() {
    local search_path="${1:-.}"
    
    print_info "Searching for DuckDB files in $search_path..."
    
    # Common locations
    local common_paths=(
        "$search_path/embeddings.duckdb"
        "$search_path/data/embeddings.duckdb"
        "./embeddings.duckdb"
    )
    
    for path in "${common_paths[@]}"; do
        if [ -f "$path" ]; then
            DUCKDB_PATH="$path"
            print_success "Found DuckDB file: $DUCKDB_PATH"
            return 0
        fi
    done
    
    # Search recursively
    local found=$(find "$search_path" -name "*.duckdb" -type f 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        DUCKDB_PATH="$found"
        print_success "Found DuckDB file: $DUCKDB_PATH"
        return 0
    fi
    
    print_error "No DuckDB file found"
    print_info "Please specify the path with --duckdb-path"
    return 1
}

# Get PostgreSQL password from .env
get_postgres_password() {
    # Use cut to get everything after the first '=' to handle passwords with '='
    POSTGRES_PASSWORD=$(grep POSTGRES_PASSWORD .env | cut -d'=' -f2-)
    
    if [ -z "$POSTGRES_PASSWORD" ]; then
        print_error "Could not read POSTGRES_PASSWORD from .env"
        exit 1
    fi
    
    # URL-encode the password to handle special characters
    # Using Python for reliable URL encoding
    POSTGRES_PASSWORD_ENCODED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$POSTGRES_PASSWORD''', safe=''))")
    
    if [ -z "$POSTGRES_PASSWORD_ENCODED" ]; then
        print_error "Failed to URL-encode password"
        exit 1
    fi
}

# Run migration
run_migration() {
    local dry_run="$1"
    local resume="$2"
    local batch_size="$3"
    local re_embed="$4"
    local embedding_model="$5"
    
    local cmd="uv run python scripts/migrate_duckdb_to_postgres.py"
    cmd="$cmd --duckdb-path $DUCKDB_PATH"
    cmd="$cmd --postgres-url postgresql://libby:${POSTGRES_PASSWORD_ENCODED}@localhost:5432/libby"
    cmd="$cmd --batch-size $batch_size"
    
    if [ "$dry_run" = "true" ]; then
        cmd="$cmd --dry-run"
    fi
    
    if [ "$resume" = "true" ]; then
        cmd="$cmd --resume"
    fi
    
    if [ "$re_embed" = "true" ]; then
        cmd="$cmd --re-embed"
    fi
    
    if [ -n "$embedding_model" ]; then
        cmd="$cmd --embedding-model $embedding_model"
    fi
    
    print_info "Running migration..."
    echo ""
    
    $cmd
}

# Main function
main() {
    local dry_run="false"
    local resume="false"
    local re_embed="false"
    local embedding_model=""
    local batch_size=1000
    local duckdb_path=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
                shift
                ;;
            --resume)
                resume="true"
                shift
                ;;
            --re-embed)
                re_embed="true"
                shift
                ;;
            --embedding-model)
                embedding_model="$2"
                shift 2
                ;;
            --batch-size)
                batch_size="$2"
                shift 2
                ;;
            --duckdb-path)
                duckdb_path="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dry-run              Preview migration without making changes"
                echo "  --resume               Resume interrupted migration"
                echo "  --re-embed             Re-embed documents if dimension mismatch detected"
                echo "  --embedding-model MODEL Embedding model for re-embedding (default: from settings)"
                echo "  --batch-size N         Number of records per batch (default: 1000)"
                echo "  --duckdb-path PATH     Path to DuckDB file (auto-detected if not specified)"
                echo "  --help                 Show this help message"
                echo ""
                echo "Examples:"
                echo "  # Preview migration"
                echo "  $0 --dry-run"
                echo ""
                echo "  # Run migration"
                echo "  $0"
                echo ""
                echo "  # Resume interrupted migration"
                echo "  $0 --resume"
                echo ""
                echo "  # Re-embed when dimension mismatch"
                echo "  $0 --re-embed"
                echo ""
                echo "  # Re-embed with specific model"
                echo "  $0 --re-embed --embedding-model mxbai-embed-large"
                echo ""
                echo "  # Fast migration with larger batches"
                echo "  $0 --batch-size 5000"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    print_banner
    
    # Run checks
    check_env
    check_postgres
    get_postgres_password
    
    # Find DuckDB file
    if [ -n "$duckdb_path" ]; then
        if [ ! -f "$duckdb_path" ]; then
            print_error "DuckDB file not found: $duckdb_path"
            exit 1
        fi
        DUCKDB_PATH="$duckdb_path"
        print_success "Using DuckDB file: $DUCKDB_PATH"
    else
        if ! find_duckdb_file; then
            exit 1
        fi
    fi
    
    # Confirm migration
    echo ""
    if [ "$dry_run" = "true" ]; then
        print_warning "DRY RUN MODE: No changes will be made"
    else
        print_warning "This will migrate data from DuckDB to PostgreSQL"
        print_info "DuckDB file: $DUCKDB_PATH"
        print_info "PostgreSQL: localhost:5432/libby"
        print_info "Batch size: $batch_size"
        echo ""
        read -p "Continue? [Y/n]: " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
            print_info "Migration cancelled"
            exit 0
        fi
    fi
    
    # Run migration
    echo ""
    run_migration "$dry_run" "$resume" "$batch_size" "$re_embed" "$embedding_model"
    
    # Post-migration steps
    if [ "$dry_run" = "false" ]; then
        echo ""
        print_info "Post-migration steps:"
        echo "  1. Verify migration: docker compose exec postgres psql -U libby -d libby -c 'SELECT COUNT(*) FROM embedding;'"
        echo "  2. Start all services: docker compose up -d"
        echo "  3. Check API health: curl http://localhost:8001/api/health"
    fi
}

# Run main function
main "$@"
