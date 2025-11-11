"""Batch document processing using LandingAI ADE Parse Jobs API.

This module provides chunked/batch processing capabilities to avoid:
1. Rate limit errors (429)
2. Authentication header issues
3. Long-running synchronous requests

Uses async Parse Jobs API for better reliability and throughput.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json
from dataclasses import dataclass
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ParseJobResult:
    """Result from a parse job."""
    job_id: str
    document_path: str
    status: str
    markdown: str = ""
    chunks: List[Dict[str, Any]] = None
    pages: int = 0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.metadata is None:
            self.metadata = {}


class ADEBatchProcessor:
    """Batch processor for LandingAI ADE Parse Jobs API.

    Features:
    - Submits documents in configurable batch sizes
    - Polls for job completion with exponential backoff
    - Handles failures gracefully with retries
    - Caches results to avoid reprocessing
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "./cache/ade_batch",
        batch_size: int = 5,
        poll_interval: int = 5,
        max_poll_attempts: int = 60
    ):
        """Initialize batch processor.

        Args:
            api_key: LandingAI API key
            cache_dir: Directory for caching parse results
            batch_size: Number of documents to submit per batch
            poll_interval: Seconds between job status polls
            max_poll_attempts: Maximum polling attempts before timeout
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

        # Initialize LandingAI SDK
        try:
            from landingai_ade import LandingAIADE  # type: ignore
            self._client = LandingAIADE(apikey=api_key)
            log.info("ade_batch_processor_initialized",
                    batch_size=batch_size,
                    api_key_length=len(api_key))
        except ImportError as e:
            raise RuntimeError(
                "LandingAI SDK is required but not installed.\n"
                "Install it with: pip install landingai-ade\n"
                f"ImportError: {str(e)}"
            ) from e

    def _get_cache_path(self, document_path: str, job_id: str) -> Path:
        """Get cache path for a job result."""
        import hashlib
        doc_hash = hashlib.md5(document_path.encode()).hexdigest()[:16]
        return self.cache_dir / f"{doc_hash}_{job_id}.json"

    def submit_parse_jobs(
        self,
        document_paths: List[str],
        split: str = "page"
    ) -> List[Dict[str, Any]]:
        """Submit multiple documents as parse jobs.

        Args:
            document_paths: List of document file paths
            split: How to split documents (page, section, etc.)

        Returns:
            List of job info dicts with job_id and document_path
        """
        jobs = []

        for doc_path in document_paths:
            try:
                log.info("submit_parse_job", document=doc_path, split=split)

                # Submit async parse job
                job = self._client.parse_jobs.create(
                    document=Path(doc_path),
                    split=split
                )

                job_info = {
                    "job_id": job.job_id,
                    "document_path": doc_path,
                    "status": "submitted",
                    "submit_time": time.time()
                }
                jobs.append(job_info)

                log.info("parse_job_submitted",
                        job_id=job.job_id,
                        document=doc_path)

                # Small delay between submissions to avoid overwhelming API
                time.sleep(0.5)

            except Exception as e:
                log.error("parse_job_submit_failed",
                         document=doc_path,
                         error=str(e))
                jobs.append({
                    "job_id": None,
                    "document_path": doc_path,
                    "status": "submit_failed",
                    "error": str(e)
                })

        return jobs

    def poll_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll a single job for status.

        Args:
            job_id: Job ID to check

        Returns:
            Job status info dict
        """
        try:
            job_status = self._client.parse_jobs.get(job_id)

            status_info = {
                "job_id": job_id,
                "status": job_status.status,
                "has_result": hasattr(job_status, 'result') and job_status.result is not None
            }

            # If completed, include result
            if job_status.status == "completed" and hasattr(job_status, 'result'):
                status_info["result"] = job_status.result

            return status_info

        except Exception as e:
            log.error("poll_job_failed", job_id=job_id, error=str(e))
            return {
                "job_id": job_id,
                "status": "poll_failed",
                "error": str(e)
            }

    def wait_for_jobs(
        self,
        jobs: List[Dict[str, Any]]
    ) -> List[ParseJobResult]:
        """Wait for all jobs to complete with polling.

        Args:
            jobs: List of job info dicts from submit_parse_jobs

        Returns:
            List of ParseJobResult objects
        """
        results = []
        pending_jobs = [j for j in jobs if j.get("job_id") is not None]

        log.info("waiting_for_jobs",
                total_jobs=len(jobs),
                pending_jobs=len(pending_jobs))

        attempts = 0
        while pending_jobs and attempts < self.max_poll_attempts:
            attempts += 1
            log.debug("poll_attempt",
                     attempt=attempts,
                     pending=len(pending_jobs))

            completed_in_round = []

            for job_info in pending_jobs:
                job_id = job_info["job_id"]
                status_info = self.poll_job_status(job_id)

                if status_info["status"] == "completed":
                    # Job finished successfully
                    result = self._process_job_result(
                        job_id,
                        job_info["document_path"],
                        status_info.get("result")
                    )
                    results.append(result)
                    completed_in_round.append(job_info)

                    log.info("job_completed",
                            job_id=job_id,
                            document=job_info["document_path"])

                elif status_info["status"] in ["failed", "poll_failed"]:
                    # Job failed
                    result = ParseJobResult(
                        job_id=job_id,
                        document_path=job_info["document_path"],
                        status="failed",
                        error=status_info.get("error", "Job failed")
                    )
                    results.append(result)
                    completed_in_round.append(job_info)

                    log.error("job_failed",
                             job_id=job_id,
                             document=job_info["document_path"],
                             error=result.error)

            # Remove completed jobs from pending
            for completed in completed_in_round:
                pending_jobs.remove(completed)

            if pending_jobs:
                # Wait before next poll
                log.debug("poll_waiting",
                         seconds=self.poll_interval,
                         pending=len(pending_jobs))
                time.sleep(self.poll_interval)

        # Handle jobs that timed out
        for job_info in pending_jobs:
            result = ParseJobResult(
                job_id=job_info["job_id"],
                document_path=job_info["document_path"],
                status="timeout",
                error=f"Job timed out after {attempts} polling attempts"
            )
            results.append(result)
            log.warning("job_timeout",
                       job_id=job_info["job_id"],
                       document=job_info["document_path"])

        return results

    def _process_job_result(
        self,
        job_id: str,
        document_path: str,
        result: Any
    ) -> ParseJobResult:
        """Process a completed job result into ParseJobResult.

        Args:
            job_id: Job ID
            document_path: Original document path
            result: Result object from job

        Returns:
            ParseJobResult object
        """
        try:
            # Extract markdown
            markdown = ""
            if hasattr(result, 'markdown'):
                markdown = result.markdown
            elif isinstance(result, dict):
                markdown = result.get('markdown', '')

            # Extract chunks
            chunks_list = []
            if hasattr(result, 'chunks') and result.chunks:
                for chunk in result.chunks:
                    if hasattr(chunk, 'model_dump'):
                        chunks_list.append(chunk.model_dump())
                    elif isinstance(chunk, dict):
                        chunks_list.append(chunk)
            elif isinstance(result, dict) and 'chunks' in result:
                chunks_list = result['chunks']

            # Extract metadata
            metadata_dict = {}
            if hasattr(result, 'metadata') and result.metadata:
                if hasattr(result.metadata, 'model_dump'):
                    metadata_dict = result.metadata.model_dump()
                elif isinstance(result.metadata, dict):
                    metadata_dict = result.metadata
            elif isinstance(result, dict) and 'metadata' in result:
                metadata_dict = result['metadata']

            # Extract pages
            pages = 0
            if hasattr(result, 'pages'):
                pages = result.pages
            elif isinstance(result, dict):
                pages = result.get('pages', 0)

            parse_result = ParseJobResult(
                job_id=job_id,
                document_path=document_path,
                status="completed",
                markdown=markdown,
                chunks=chunks_list,
                pages=pages,
                metadata=metadata_dict
            )

            # Cache result
            cache_path = self._get_cache_path(document_path, job_id)
            self._save_to_cache(cache_path, parse_result)

            return parse_result

        except Exception as e:
            log.error("process_result_failed",
                     job_id=job_id,
                     error=str(e))
            return ParseJobResult(
                job_id=job_id,
                document_path=document_path,
                status="processing_failed",
                error=str(e)
            )

    def _save_to_cache(self, cache_path: Path, result: ParseJobResult):
        """Save parse result to cache."""
        try:
            cache_data = {
                "job_id": result.job_id,
                "document_path": result.document_path,
                "status": result.status,
                "markdown": result.markdown,
                "chunks": result.chunks,
                "pages": result.pages,
                "metadata": result.metadata,
                "error": result.error
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

            log.debug("result_cached", cache_path=str(cache_path))

        except Exception as e:
            log.warning("cache_save_failed",
                       cache_path=str(cache_path),
                       error=str(e))

    def process_documents_in_batches(
        self,
        document_paths: List[str],
        split: str = "page"
    ) -> List[ParseJobResult]:
        """Process multiple documents in batches.

        This is the main entry point for batch processing.

        Args:
            document_paths: List of document file paths
            split: How to split documents

        Returns:
            List of ParseJobResult objects for all documents
        """
        all_results = []
        total_docs = len(document_paths)

        log.info("batch_processing_started",
                total_documents=total_docs,
                batch_size=self.batch_size)

        # Process in batches
        for i in range(0, total_docs, self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_docs = document_paths[i:i + self.batch_size]

            log.info("processing_batch",
                    batch_num=batch_num,
                    batch_size=len(batch_docs),
                    total_batches=(total_docs + self.batch_size - 1) // self.batch_size)

            # Submit jobs for this batch
            jobs = self.submit_parse_jobs(batch_docs, split=split)

            # Wait for batch to complete
            batch_results = self.wait_for_jobs(jobs)
            all_results.extend(batch_results)

            log.info("batch_completed",
                    batch_num=batch_num,
                    completed=len([r for r in batch_results if r.status == "completed"]),
                    failed=len([r for r in batch_results if r.status != "completed"]))

            # Add delay between batches to avoid rate limits
            if i + self.batch_size < total_docs:
                batch_delay = 10.0
                log.info("batch_delay",
                        seconds=batch_delay,
                        reason="rate_limit_prevention")
                time.sleep(batch_delay)

        # Summary
        completed = len([r for r in all_results if r.status == "completed"])
        failed = len([r for r in all_results if r.status != "completed"])

        log.info("batch_processing_finished",
                total_documents=total_docs,
                completed=completed,
                failed=failed,
                success_rate=f"{(completed/total_docs)*100:.1f}%")

        return all_results
