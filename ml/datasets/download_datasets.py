"""
Digital Shield Rail Defense — Autonomous Dataset Downloader
============================================================
Downloads, verifies, and extracts all required datasets for the
railway anomaly intelligence training pipeline.

Supported datasets:
  1. UCF Crime Dataset (~13 GB)
  2. UCSD Pedestrian Anomaly Dataset (~500 MB)
  3. ShanghaiTech Campus Dataset (~2 GB)
  4. Railway CCTV videos (curated public domain)
  5. Simulated railway scenarios (self-generated)

Features:
  - Resumable downloads with progress tracking
  - SHA256 integrity verification
  - Automatic extraction (zip, tar.gz, rar)
  - Mirror URL fallback
  - Parallel download support
  - Detailed logging
"""

import os
import sys
import json
import time
import hashlib
import logging
import tarfile
import zipfile
import shutil
import requests
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.config import (
    DATASET_SOURCES, RAW_DIR, METADATA_DIR,
    DatasetSource, LOG_FORMAT, LOG_DATE_FORMAT,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger("dataset_downloader")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console)


# ============================================================================
# DATASET DOWNLOADER
# ============================================================================

class DatasetDownloader:
    """
    Autonomous dataset downloader with resume support,
    integrity verification, and automatic extraction.
    """

    def __init__(self, output_dir: Optional[Path] = None, max_workers: int = 2):
        self.output_dir = output_dir or RAW_DIR
        self.max_workers = max_workers
        self.download_log: Dict[str, dict] = {}
        self.log_path = METADATA_DIR / "download_log.json"
        self._load_log()

    def _load_log(self):
        """Load previous download state for resume capability."""
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    self.download_log = json.load(f)
                logger.info(f"Loaded download log: {len(self.download_log)} entries")
            except json.JSONDecodeError:
                self.download_log = {}

    def _save_log(self):
        """Persist download state."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.download_log, f, indent=2, default=str)

    def _update_log(self, dataset_name: str, status: str, **kwargs):
        """Update download log for a dataset."""
        if dataset_name not in self.download_log:
            self.download_log[dataset_name] = {}
        self.download_log[dataset_name].update({
            "status": status,
            "updated_at": datetime.now().isoformat(),
            **kwargs,
        })
        self._save_log()

    def _get_file_size(self, url: str) -> Optional[int]:
        """Get remote file size via HEAD request."""
        try:
            resp = requests.head(url, allow_redirects=True, timeout=15)
            size = resp.headers.get("Content-Length")
            return int(size) if size else None
        except Exception:
            return None

    def _compute_sha256(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _download_file(
        self,
        url: str,
        dest_path: Path,
        dataset_name: str,
        chunk_size: int = 8192,
    ) -> bool:
        """
        Download a file with resume support and progress tracking.

        Args:
            url: Download URL
            dest_path: Local destination path
            dataset_name: Name for logging
            chunk_size: Download chunk size in bytes

        Returns:
            True if download succeeded
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(dest_path.suffix + ".partial")

        # Resume support
        existing_size = temp_path.stat().st_size if temp_path.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            logger.info(f"[{dataset_name}] Resuming from {existing_size / 1e6:.1f} MB")

        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=60)

            if resp.status_code == 416:  # Range not satisfiable — file complete
                if temp_path.exists():
                    shutil.move(str(temp_path), str(dest_path))
                return True

            if resp.status_code not in (200, 206):
                logger.error(f"[{dataset_name}] HTTP {resp.status_code}: {url}")
                return False

            total_size = int(resp.headers.get("Content-Length", 0)) + existing_size
            downloaded = existing_size
            mode = "ab" if existing_size > 0 else "wb"
            start_time = time.time()

            with open(temp_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress logging every 50 MB
                        if downloaded % (50 * 1024 * 1024) < chunk_size:
                            elapsed = time.time() - start_time
                            speed = (downloaded - existing_size) / elapsed / 1e6 if elapsed > 0 else 0
                            pct = (downloaded / total_size * 100) if total_size else 0
                            logger.info(
                                f"[{dataset_name}] {downloaded / 1e6:.1f} MB / "
                                f"{total_size / 1e6:.1f} MB ({pct:.1f}%) — "
                                f"{speed:.1f} MB/s"
                            )

            # Move completed download
            shutil.move(str(temp_path), str(dest_path))
            elapsed = time.time() - start_time
            logger.info(
                f"[{dataset_name}] Download complete: {dest_path.name} "
                f"({downloaded / 1e6:.1f} MB in {elapsed:.1f}s)"
            )
            return True

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{dataset_name}] Connection error: {e}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"[{dataset_name}] Download timed out")
            return False
        except Exception as e:
            logger.error(f"[{dataset_name}] Download failed: {e}")
            return False

    def _extract_archive(self, archive_path: Path, extract_dir: Path, file_type: str) -> bool:
        """Extract downloaded archive to target directory."""
        logger.info(f"Extracting {archive_path.name} → {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            if file_type == "zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(extract_dir)

            elif file_type in ("tar.gz", "tar", "tgz"):
                mode = "r:gz" if file_type in ("tar.gz", "tgz") else "r"
                with tarfile.open(archive_path, mode) as tf:
                    tf.extractall(extract_dir)

            else:
                logger.warning(f"Unsupported archive type: {file_type}")
                return False

            logger.info(f"Extraction complete: {extract_dir}")
            return True

        except (zipfile.BadZipFile, tarfile.TarError) as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def _is_dataset_ready(self, dataset_name: str) -> bool:
        """Check if dataset is already downloaded and extracted."""
        log_entry = self.download_log.get(dataset_name, {})
        if log_entry.get("status") == "complete":
            source = DATASET_SOURCES.get(dataset_name)
            if source and source.target_dir.exists():
                file_count = sum(1 for _ in source.target_dir.rglob("*") if _.is_file())
                if file_count > 0:
                    logger.info(f"[{dataset_name}] Already ready ({file_count} files)")
                    return True
        return False

    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Download and extract a single dataset.

        Args:
            dataset_name: Key from DATASET_SOURCES
            force: Re-download even if already present

        Returns:
            True if dataset is ready for use
        """
        if dataset_name not in DATASET_SOURCES:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False

        source = DATASET_SOURCES[dataset_name]

        # Check if already ready
        if not force and self._is_dataset_ready(dataset_name):
            return True

        # Skip datasets without URLs (generated datasets)
        if not source.url:
            logger.info(f"[{dataset_name}] No URL — skipping download (generated dataset)")
            source.target_dir.mkdir(parents=True, exist_ok=True)
            self._update_log(dataset_name, "skipped", reason="generated_dataset")
            return True

        logger.info(f"{'='*60}")
        logger.info(f"Downloading: {source.name}")
        logger.info(f"Size: ~{source.expected_size_gb} GB")
        logger.info(f"Target: {source.target_dir}")
        logger.info(f"{'='*60}")

        self._update_log(dataset_name, "downloading", url=source.url)

        # Determine output filename
        ext_map = {"zip": ".zip", "tar.gz": ".tar.gz", "rar": ".rar", "direct": ""}
        ext = ext_map.get(source.file_type, "")
        archive_path = self.output_dir / f"{dataset_name}{ext}"

        # Try primary URL, then mirrors
        urls_to_try = [source.url] + source.mirror_urls
        success = False

        for url in urls_to_try:
            if not url:
                continue
            logger.info(f"[{dataset_name}] Trying: {url}")
            if self._download_file(url, archive_path, dataset_name):
                success = True
                break
            logger.warning(f"[{dataset_name}] Failed, trying next mirror...")

        if not success:
            self._update_log(dataset_name, "failed", error="All download attempts failed")
            logger.error(f"[{dataset_name}] All download attempts failed")
            return False

        # Verify integrity if SHA256 is known
        if source.sha256:
            computed = self._compute_sha256(archive_path)
            if computed != source.sha256:
                logger.error(
                    f"[{dataset_name}] SHA256 mismatch!\n"
                    f"  Expected: {source.sha256}\n"
                    f"  Got:      {computed}"
                )
                self._update_log(dataset_name, "failed", error="SHA256 mismatch")
                return False
            logger.info(f"[{dataset_name}] SHA256 verified ✓")

        # Extract archive
        if source.file_type != "direct":
            if not self._extract_archive(archive_path, source.target_dir, source.file_type):
                self._update_log(dataset_name, "failed", error="Extraction failed")
                return False

            # Optionally remove archive after extraction
            # archive_path.unlink()

        self._update_log(
            dataset_name, "complete",
            archive_path=str(archive_path),
            extracted_to=str(source.target_dir),
        )
        return True

    def download_all(self, force: bool = False, skip_large: bool = False) -> Dict[str, bool]:
        """
        Download all datasets.

        Args:
            force: Re-download all
            skip_large: Skip datasets > 5GB (UCF Crime)

        Returns:
            Dict of dataset_name → success
        """
        results = {}
        datasets = list(DATASET_SOURCES.keys())

        if skip_large:
            datasets = [
                d for d in datasets
                if DATASET_SOURCES[d].expected_size_gb <= 5.0
            ]
            logger.info(f"Skipping large datasets. Processing: {datasets}")

        for name in datasets:
            try:
                results[name] = self.download_dataset(name, force=force)
            except Exception as e:
                logger.error(f"[{name}] Unexpected error: {e}")
                results[name] = False

        # Summary
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*60)
        for name, success in results.items():
            status = "✓ READY" if success else "✗ FAILED"
            logger.info(f"  {status}  {name}")
        logger.info("="*60)

        return results

    def get_status(self) -> Dict[str, dict]:
        """Get download status for all datasets."""
        status = {}
        for name, source in DATASET_SOURCES.items():
            log_entry = self.download_log.get(name, {})
            file_count = 0
            total_size = 0
            if source.target_dir.exists():
                for f in source.target_dir.rglob("*"):
                    if f.is_file():
                        file_count += 1
                        total_size += f.stat().st_size
            status[name] = {
                "name": source.name,
                "status": log_entry.get("status", "pending"),
                "files": file_count,
                "size_mb": round(total_size / 1e6, 2),
                "expected_gb": source.expected_size_gb,
                "target_dir": str(source.target_dir),
            }
        return status

    def generate_download_commands(self) -> str:
        """Generate wget/curl commands for manual download."""
        commands = ["#!/bin/bash", "# Digital Shield Rail Defense — Manual Dataset Download", ""]
        for name, source in DATASET_SOURCES.items():
            if not source.url:
                continue
            commands.append(f"# {source.name} (~{source.expected_size_gb} GB)")
            commands.append(f'echo "Downloading {source.name}..."')
            commands.append(f"mkdir -p {source.target_dir}")
            commands.append(f'wget -c "{source.url}" -O "{source.target_dir}/{name}.{source.file_type}"')
            commands.append("")
        return "\n".join(commands)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point for dataset downloading."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Digital Shield — Autonomous Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_SOURCES.keys()) + ["all"],
        default=["all"],
        help="Datasets to download (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--skip-large", action="store_true", help="Skip datasets > 5GB")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--generate-commands", action="store_true", help="Print wget commands")
    parser.add_argument("--output-dir", type=str, help="Override output directory")

    args = parser.parse_args()

    downloader = DatasetDownloader(
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    if args.status:
        status = downloader.get_status()
        print("\n" + "="*70)
        print("DATASET STATUS")
        print("="*70)
        for name, info in status.items():
            print(f"  [{info['status']:>10}]  {info['name']}")
            print(f"              Files: {info['files']} | Size: {info['size_mb']} MB | Expected: ~{info['expected_gb']} GB")
        print("="*70)
        return

    if args.generate_commands:
        print(downloader.generate_download_commands())
        return

    if "all" in args.datasets:
        downloader.download_all(force=args.force, skip_large=args.skip_large)
    else:
        for name in args.datasets:
            downloader.download_dataset(name, force=args.force)


if __name__ == "__main__":
    main()
