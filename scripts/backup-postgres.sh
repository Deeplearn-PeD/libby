#!/bin/sh
set -e

echo "Starting PostgreSQL backup service..."

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/libby_backup_${TIMESTAMP}.sql.gz"

create_backup() {
    echo "Creating backup: $BACKUP_FILE"
    pg_dump -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" | gzip > "$BACKUP_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Backup created successfully: $BACKUP_FILE"
        
        BACKUP_SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
        echo "Backup size: $BACKUP_SIZE"
    else
        echo "ERROR: Backup failed!"
        rm -f "$BACKUP_FILE"
        return 1
    fi
}

cleanup_old_backups() {
    echo "Cleaning up backups older than $BACKUP_RETENTION_DAYS days..."
    find "$BACKUP_DIR" -name "libby_backup_*.sql.gz" -type f -mtime +$BACKUP_RETENTION_DAYS -delete
    echo "Cleanup complete"
}

list_backups() {
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"/libby_backup_*.sql.gz 2>/dev/null || echo "No backups found"
}

if [ "$1" = "--list" ]; then
    list_backups
    exit 0
fi

if [ "$1" = "--manual" ]; then
    create_backup
    exit $?
fi

echo "Starting automated backup scheduler..."
echo "Backup schedule: $BACKUP_SCHEDULE"
echo "Backup retention: $BACKUP_RETENTION_DAYS days"

while true; do
    CURRENT_TIME=$(date +%H%M)
    SCHEDULE_TIME=$(echo "$BACKUP_SCHEDULE" | awk '{printf "%02d%02d", $2, $1}')
    
    if [ "$CURRENT_TIME" = "$SCHEDULE_TIME" ]; then
        echo "Scheduled backup time reached: $(date)"
        create_backup
        cleanup_old_backups
        echo "Next backup scheduled for tomorrow at $BACKUP_SCHEDULE"
        sleep 86400
    else
        sleep 60
    fi
done
