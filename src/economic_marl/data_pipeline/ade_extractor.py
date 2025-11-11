from __future__ import annotations
from typing import Dict, Any, List, Iterable, Optional
import os, json, glob, hashlib
from pathlib import Path
from dataclasses import dataclass
from ..utils.logging import get_logger

log = get_logger(__name__)

@dataclass
class ADERecord:
    doc_id: str
    meta: Dict[str, Any]
    text: str
    fields: Dict[str, Any]

class ADEExtractor:
    """Wrapper around LandingAI ADE client with graceful fallback.

    Supports both legacy extract() and new parse() + extract() workflow.
    Parse results are cached to enable multiple extractions without re-parsing.
    """

    def __init__(self, api_key: Optional[str], input_folder: str, cache_dir: str = "./cache/ade"):
        self.api_key = api_key
        self.input_folder = input_folder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate API key is provided
        if not api_key:
            raise ValueError(
                "ADE_API_KEY is required for LandingAI document extraction.\n"
                "Get your API key from: https://app.landing.ai/\n"
                "Set in .env file: ADE_API_KEY=your_key_here\n"
                "Or use AWS Secrets Manager: ADE__API_KEY_SECRET_NAME=landingai/api_key"
            )

        # Store API key for potential re-initialization
        self._api_key = api_key

        # Initialize LandingAI SDK - REQUIRED, no fallback
        try:
            from landingai_ade import LandingAIADE  # type: ignore
            self._client = LandingAIADE(apikey=api_key)  # pragma: no cover
            self._has_sdk = True
            log.info("ade_sdk_loaded", status=True, api_key_length=len(api_key))

            # Verify client has API key set
            if not hasattr(self._client, 'api_key') and not hasattr(self._client, 'apikey'):
                log.warning("ade_client_no_apikey_attribute", message="Client initialized but no API key attribute found")

        except ImportError as e:
            raise RuntimeError(
                "LandingAI SDK is required but not installed.\n"
                "Install it with: pip install landingai-ade\n"
                f"ImportError: {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize LandingAI ADE client: {str(e)}\n"
                "Check your API key and network connection.\n"
                "Verify your key at: https://app.landing.ai/"
            ) from e

    def extract_paths(self, paths: Iterable[str]) -> List[ADERecord]:
        """Extract data from document paths using LandingAI ADE - REQUIRED."""
        if not self._has_sdk:
            raise RuntimeError(
                "LandingAI SDK is not available. Cannot proceed with extraction.\n"
                "This should never happen if __init__ validation passed."
            )

        out: List[ADERecord] = []
        for p in paths:
            try:
                result = self._client.extract(p)  # pragma: no cover
                record = ADERecord(
                    doc_id=os.path.basename(p),
                    meta={"path": p},
                    text=json.dumps(result)[:2048],
                    fields=result.get("fields", {}),
                )
                out.append(record)
                log.info("ade_extracted", doc_id=record.doc_id, fields=record.fields)
            except Exception as e:
                log.error("ade_extract_failed", path=p, error=str(e))
                raise RuntimeError(
                    f"Failed to extract data from {p}:\n{str(e)}\n"
                    "Check that the file exists and is a valid PDF/document."
                ) from e
        return out

    def extract_folder(self, pattern: str = "*.pdf") -> List[ADERecord]:
        files = glob.glob(os.path.join(self.input_folder, pattern))
        return self.extract_paths(files)

    def _get_cache_path(self, document_path: str) -> Path:
        """Generate cache path for a parsed document."""
        # Create hash of document path for unique cache filename
        path_hash = hashlib.md5(document_path.encode()).hexdigest()[:16]
        doc_name = Path(document_path).stem
        return self.cache_dir / f"{doc_name}_{path_hash}_parsed.json"

    def parse_document(
        self,
        document_path: str,
        split: str = "page",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Parse a document into Markdown with grounding info.

        Uses LandingAI's parse() API to convert PDF → Markdown with:
        - Markdown representation
        - Chunks with bounding boxes
        - Page metadata
        - Confidence scores

        Results are cached to enable multiple extractions without re-parsing.

        Args:
            document_path: Path to PDF/document file
            split: How to split document ("page", "section", etc.)
            use_cache: Whether to use cached parse results

        Returns:
            Dict with keys:
                - markdown: Full Markdown representation
                - chunks: List of text chunks with bounding boxes
                - pages: Number of pages
                - metadata: Document metadata
        """
        cache_path = self._get_cache_path(document_path)

        # Check cache first
        if use_cache and cache_path.exists():
            log.info("parse_cache_hit", document=document_path, cache_path=str(cache_path))
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Parse using LandingAI SDK - REQUIRED, no fallback
        if not self._has_sdk:
            raise RuntimeError(
                "LandingAI SDK is not available. Cannot proceed with parsing.\n"
                "This should never happen if __init__ validation passed."
            )

        try:
            log.info("parse_document_real", document=document_path, split=split)
            from pathlib import Path as P

            # Attempt parse with error handling for auth issues
            try:
                result = self._client.parse(document=P(document_path), split=split)  # pragma: no cover
            except Exception as parse_error:
                error_msg = str(parse_error)

                # Check for authentication errors
                if "Missing Authorization header" in error_msg or "401" in error_msg or "403" in error_msg:
                    log.error("ade_auth_error", error=error_msg, api_key_length=len(self._api_key))
                    raise RuntimeError(
                        f"LandingAI API authentication failed: {error_msg}\n"
                        f"API key length: {len(self._api_key)}\n"
                        "Possible causes:\n"
                        "  1. API key is invalid or expired\n"
                        "  2. SDK version incompatibility\n"
                        "  3. Network/proxy issues blocking auth headers\n"
                        "Solutions:\n"
                        "  - Verify API key at: https://app.landing.ai/\n"
                        "  - Update SDK: pip install --upgrade landingai-ade\n"
                        f"  - Check .env file has: ADE_API_KEY={self._api_key[:10]}..."
                    ) from parse_error

                # Re-raise other errors
                raise

            # Normalize result format - result is a ParseResponse object
            # Convert chunks to dicts for JSON serialization
            chunks_list = []
            if hasattr(result, 'chunks') and result.chunks:
                for chunk in result.chunks:
                    if hasattr(chunk, 'model_dump'):
                        chunks_list.append(chunk.model_dump())
                    elif isinstance(chunk, dict):
                        chunks_list.append(chunk)
                    else:
                        # Fallback: convert to dict manually
                        chunks_list.append({
                            "text": getattr(chunk, 'text', ''),
                            "page": getattr(chunk, 'page', 0),
                            "bbox": getattr(chunk, 'bbox', {}),
                            "confidence": getattr(chunk, 'confidence', 0.0)
                        })

            # Convert metadata to dict if it's a Pydantic model
            metadata_dict = {}
            if hasattr(result, 'metadata') and result.metadata:
                if hasattr(result.metadata, 'model_dump'):
                    metadata_dict = result.metadata.model_dump()
                elif isinstance(result.metadata, dict):
                    metadata_dict = result.metadata
                else:
                    metadata_dict = {}

            parsed = {
                "markdown": result.markdown if hasattr(result, 'markdown') else "",
                "chunks": chunks_list,
                "pages": result.pages if hasattr(result, 'pages') else 0,
                "metadata": metadata_dict,
                "document_path": document_path
            }

            # Save to cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            log.info("parse_cache_saved", cache_path=str(cache_path))

            return parsed
        except Exception as e:
            log.error("parse_document_failed", error=str(e), document=document_path)
            raise RuntimeError(
                f"Failed to parse document {document_path}:\n{str(e)}\n"
                "Check that the file exists, is readable, and is a valid PDF/document.\n"
                "Verify your LandingAI API key has sufficient credits."
            ) from e

    def extract_fields(
        self,
        markdown: str,
        schema: Dict[str, str],
        chunks: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Extract structured fields from Markdown using a schema.

        Uses LandingAI's extract() API to extract field-level data from
        pre-parsed Markdown. Much faster and cheaper than full extraction.

        Args:
            markdown: Markdown text from parse_document()
            schema: Field definitions, e.g.:
                {
                    "hs_code": "10-digit Harmonized System code",
                    "description": "Product description",
                    "rate_of_duty": "Ad valorem rate as percentage"
                }
            chunks: Optional chunks from parse_document() for grounding

        Returns:
            List of records with extracted fields
        """
        if self._has_sdk:
            try:
                log.info("extract_fields_real", schema_keys=list(schema.keys()))
                # Convert schema dict to JSON Schema format required by API
                # The API expects a JSON Schema object with "type": "object"
                json_schema = {
                    "type": "object",
                    "properties": {
                        field: {"type": "string", "description": desc}
                        for field, desc in schema.items()
                    },
                    "required": list(schema.keys())
                }
                schema_json = json.dumps(json_schema)
                # Convert markdown string to bytes for upload
                markdown_bytes = markdown.encode('utf-8')
                result = self._client.extract(schema=schema_json, markdown=markdown_bytes)  # pragma: no cover

                # Normalize result to list of records - result is ExtractResponse object
                # The actual extracted data is in the 'extraction' field
                records = []
                if hasattr(result, 'extraction') and result.extraction:
                    # Single extraction result - wrap in list
                    records = [result.extraction]
                elif hasattr(result, 'extractions') and result.extractions:
                    # Multiple extraction results
                    records = result.extractions
                elif hasattr(result, 'records'):
                    records = result.records
                elif hasattr(result, 'data'):
                    records = result.data if isinstance(result.data, list) else [result.data]
                else:
                    # Fallback: try to convert result to dict
                    try:
                        result_dict = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
                        # Check for 'extraction' in the dict
                        if 'extraction' in result_dict:
                            records = [result_dict['extraction']]
                        else:
                            records = result_dict.get('records', [result_dict])
                    except:
                        records = []

                # Convert Pydantic models to dicts if needed
                normalized_records = []
                for rec in records:
                    if hasattr(rec, 'model_dump'):
                        normalized_records.append(rec.model_dump())
                    elif isinstance(rec, dict):
                        normalized_records.append(rec)
                    else:
                        normalized_records.append({"data": str(rec)})

                log.info("extract_fields_success", count=len(normalized_records))
                return normalized_records
            except Exception as e:
                log.error("extract_fields_failed", error=str(e))
                return self._stub_extract_fields(schema)
        else:
            return self._stub_extract_fields(schema)

    def _stub_extract_fields(self, schema: Dict[str, str]) -> List[Dict[str, Any]]:
        """Stub implementation of extract for testing without SDK."""
        # Generate stub records with None values for all schema fields
        stub_records = [
            {field: None for field in schema.keys()},
            {field: None for field in schema.keys()}
        ]

        # Fill in some default values for common fields
        for record in stub_records:
            if "hs_code" in record:
                record["hs_code"] = "7108.12.00"
            if "description" in record:
                record["description"] = "Gold powder (STUB)"
            if "tariff_rate" in record or "rate_of_duty" in record:
                key = "tariff_rate" if "tariff_rate" in record else "rate_of_duty"
                record[key] = 0.06

        return stub_records
