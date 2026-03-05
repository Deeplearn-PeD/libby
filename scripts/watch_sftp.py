#!/usr/bin/env python3
"""
SFTP Document Watcher for Libby D. Bot

Monitors an SFTP upload directory for new PDF files and submits them
to the Libby API for embedding.
"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger


class DocumentWatcher:
    def __init__(
        self,
        watch_dir: str,
        api_url: str,
        collection_name: str = "main",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        processed_dir_name: str = "processed",
        failed_dir_name: str = "failed",
    ):
        self.watch_dir = Path(watch_dir)
        self.api_url = api_url.rstrip("/")
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_dir = self.watch_dir / processed_dir_name
        self.failed_dir = self.watch_dir / failed_dir_name

        self._ensure_directories()

    def _ensure_directories(self):
        """Create processed and failed directories if they don't exist."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Watch directory: {self.watch_dir}")
        logger.info(f"Processed directory: {self.processed_dir}")
        logger.info(f"Failed directory: {self.failed_dir}")

    def _get_pdf_files(self) -> list[Path]:
        """Get list of PDF files in watch directory (excluding subdirectories)."""
        pdf_files = []
        for item in self.watch_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".pdf":
                pdf_files.append(item)
        return sorted(pdf_files)

    def _embed_file(self, file_path: Path) -> tuple[bool, str]:
        """
        Submit a PDF file to the Libby API for embedding.

        Returns:
            Tuple of (success: bool, message: str)
        """
        url = f"{self.api_url}/api/embed/upload"

        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/pdf")}
                data = {
                    "collection_name": self.collection_name,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }

                logger.info(f"Submitting {file_path.name} to Libby API...")
                response = requests.post(url, files=files, data=data, timeout=300)

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get("chunks_embedded", 0)
                    return True, f"Successfully embedded {chunks} chunks"
                else:
                    error_detail = response.json().get("detail", response.text)
                    return False, f"API error: {error_detail}"

        except requests.exceptions.Timeout:
            return False, "Request timed out"
        except requests.exceptions.ConnectionError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def _move_file(self, file_path: Path, target_dir: Path, suffix: str = "") -> Path:
        """
        Move file to target directory with optional suffix.

        Returns:
            Path to the moved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = file_path.stem
        new_name = f"{stem}_{timestamp}{suffix}{file_path.suffix}"
        target_path = target_dir / new_name
        shutil.move(str(file_path), str(target_path))
        return target_path

    def process_files(self) -> dict:
        """
        Process all PDF files in the watch directory.

        Returns:
            Statistics about processing
        """
        stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "files": [],
        }

        pdf_files = self._get_pdf_files()
        stats["total"] = len(pdf_files)

        if not pdf_files:
            logger.info("No PDF files found in watch directory")
            return stats

        logger.info(f"Found {len(pdf_files)} PDF file(s) to process")

        for file_path in pdf_files:
            logger.info(f"Processing: {file_path.name}")
            success, message = self._embed_file(file_path)

            file_info = {
                "name": file_path.name,
                "success": success,
                "message": message,
            }
            stats["files"].append(file_info)

            if success:
                logger.info(f"✓ {file_path.name}: {message}")
                moved_path = self._move_file(file_path, self.processed_dir)
                logger.info(f"  Moved to: {moved_path}")
                stats["processed"] += 1
            else:
                logger.error(f"✗ {file_path.name}: {message}")
                moved_path = self._move_file(file_path, self.failed_dir)
                logger.info(f"  Moved to: {moved_path}")
                stats["failed"] += 1

        logger.info(
            f"Processing complete: {stats['processed']} processed, "
            f"{stats['failed']} failed, {stats['skipped']} skipped"
        )

        return stats

    def wait_for_api(self, max_retries: int = 30, retry_interval: int = 10) -> bool:
        """
        Wait for the Libby API to become available.

        Returns:
            True if API is available, False otherwise
        """
        health_url = f"{self.api_url}/api/health"

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Libby API is available (attempt {attempt})")
                    return True
            except requests.exceptions.RequestException:
                pass

            if attempt < max_retries:
                logger.info(
                    f"Waiting for Libby API... (attempt {attempt}/{max_retries})"
                )
                time.sleep(retry_interval)

        logger.error(f"Libby API not available after {max_retries} attempts")
        return False


def main():
    watch_dir = os.getenv("WATCH_DIR", "/data/uploads")
    api_url = os.getenv("LIBBY_API_URL", "http://libby-api:8000")
    collection_name = os.getenv("COLLECTION_NAME", "main")
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    logger.info("=" * 60)
    logger.info("Libby Document Watcher")
    logger.info("=" * 60)
    logger.info(f"Watch directory: {watch_dir}")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Chunk overlap: {chunk_overlap}")

    watcher = DocumentWatcher(
        watch_dir=watch_dir,
        api_url=api_url,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not watcher.wait_for_api():
        logger.error("Exiting: Libby API is not available")
        exit(1)

    stats = watcher.process_files()

    if stats["failed"] > 0:
        exit(1)

    exit(0)


if __name__ == "__main__":
    main()
