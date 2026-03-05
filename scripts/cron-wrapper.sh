#!/bin/sh
set -e

CRON_SCHEDULE="${CRON_SCHEDULE:-*/5 * * * *}"

echo "Starting Libby Document Watcher Cron"
echo "Schedule: $CRON_SCHEDULE"
echo "Watch directory: ${WATCH_DIR:-/data/uploads}"
echo "API URL: ${LIBBY_API_URL:-http://libby-api:8000}"

echo "$CRON_SCHEDULE cd /app && /usr/local/bin/python /app/watch_sftp.py >> /var/log/watcher.log 2>&1" | crontab -

echo "Cron job installed. Starting cron daemon..."

exec crond -f -l 2
