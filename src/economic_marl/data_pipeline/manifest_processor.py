"""
Manifest-driven document processing for LandingAI ADE + AWS Bedrock
Reads ingestion_manifest.yaml and orchestrates extraction pipeline
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from ..utils.logging import get_logger
from .ade_extractor import ADEExtractor
from .llm_analyzer import LLMAnalyzer
from .document_loader import to_parquet

log = get_logger(__name__)


@dataclass
class DocumentConfig:
    """Configuration for a single document to process"""
    name: str
    url: str
    doc_type: str
    extraction_mode: str
    fields: List[str]
    ade_template: str
    priority: str
    llm_analysis: bool = False
    purpose: Optional[str] = None
    date: Optional[str] = None


class ManifestProcessor:
    """
    Process documents according to YAML manifest configuration

    Usage:
        processor = ManifestProcessor("data/ingestion_manifest.yaml")
        processor.process_all()
    """

    def __init__(
        self,
        manifest_path: str,
        ade_key: Optional[str] = None,
        bedrock_model: Optional[str] = None,
        bedrock_region: Optional[str] = None
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()

        # Initialize extractors
        self.ade_key = ade_key
        self.bedrock_model = bedrock_model or self.manifest.get('metadata', {}).get('bedrock_model')
        self.bedrock_region = bedrock_region or self.manifest.get('metadata', {}).get('bedrock_region')

        log.info(
            "manifest_processor_initialized",
            manifest=str(self.manifest_path),
            bedrock_model=self.bedrock_model
        )

    def _load_manifest(self) -> Dict[str, Any]:
        """Load and parse YAML manifest"""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path) as f:
            manifest = yaml.safe_load(f)

        log.info("manifest_loaded", path=str(self.manifest_path))
        return manifest

    def _get_all_documents(self) -> List[DocumentConfig]:
        """Extract all document configurations from manifest"""
        documents = []

        # Process each category
        for category_name, category_data in self.manifest.items():
            if category_name in ['metadata', 'ade_templates', 'llm_analysis', 'pipeline', 'schedule']:
                continue

            # Navigate nested structure
            if isinstance(category_data, dict):
                for source_name, source_data in category_data.items():
                    if isinstance(source_data, dict) and 'documents' in source_data:
                        priority = source_data.get('priority', 'medium')

                        for doc in source_data['documents']:
                            doc_config = DocumentConfig(
                                name=doc['name'],
                                url=doc.get('url', ''),
                                doc_type=doc.get('doc_type', 'pdf'),
                                extraction_mode=doc.get('extraction_mode', 'table'),
                                fields=doc.get('fields', []),
                                ade_template=doc.get('ade_template', 'default'),
                                priority=priority,
                                llm_analysis=doc.get('llm_analysis', False),
                                purpose=doc.get('purpose'),
                                date=doc.get('date')
                            )
                            documents.append(doc_config)

        log.info("documents_collected", count=len(documents))
        return documents

    def process_document(
        self,
        doc_config: DocumentConfig,
        input_folder: str,
        ade_extractor: Optional[ADEExtractor] = None,
        llm_analyzer: Optional[LLMAnalyzer] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single document using parse + extract workflow

        Workflow:
        1. Parse document once (cached)
        2. Extract fields using schema
        3. Optionally enrich with LLM analysis

        Args:
            doc_config: Document configuration
            input_folder: Folder containing downloaded documents
            ade_extractor: ADE extractor instance (optional)
            llm_analyzer: LLM analyzer instance (optional)

        Returns:
            List of extracted records (one per row in document)
        """
        log.info(
            "processing_document",
            name=doc_config.name,
            priority=doc_config.priority,
            use_parse_extract=ade_extractor is not None
        )

        # Find document file
        doc_filename = self._get_document_filename(doc_config, input_folder)
        if not doc_filename:
            log.warning("document_file_not_found", name=doc_config.name)
            return []

        # Step 1: Parse document (cached)
        if ade_extractor is not None:
            parsed = ade_extractor.parse_document(
                document_path=doc_filename,
                split="page",
                use_cache=True
            )
            log.info("document_parsed", name=doc_config.name, pages=parsed.get('pages', 0))

            # Step 2: Build extraction schema from fields
            schema = self._build_schema(doc_config)

            # Step 3: Extract fields
            extracted_records = ade_extractor.extract_fields(
                markdown=parsed['markdown'],
                schema=schema,
                chunks=parsed.get('chunks')
            )

            # Step 4: Add metadata to each record
            enriched_records = []
            for record in extracted_records:
                enriched = {
                    "doc_id": doc_config.name,
                    "url": doc_config.url,
                    "doc_type": doc_config.doc_type,
                    "extraction_mode": doc_config.extraction_mode,
                    "priority": doc_config.priority,
                    "status": "parsed_and_extracted",
                    **record
                }

                # Step 5: LLM enrichment if enabled
                if doc_config.llm_analysis and llm_analyzer is not None:
                    try:
                        llm_results = llm_analyzer.analyze(
                            text=parsed['markdown'][:4000],  # Limit to first 4000 chars
                            prompt_type="tariff_sentiment"
                        )
                        enriched.update(llm_results)
                        enriched["status"] = "parsed_extracted_llm_enriched"
                    except Exception as e:
                        log.error("llm_enrichment_failed", doc=doc_config.name, error=str(e))

                enriched_records.append(enriched)

            return enriched_records
        else:
            # Fallback: return stub structure
            record = {
                "doc_id": doc_config.name,
                "url": doc_config.url,
                "doc_type": doc_config.doc_type,
                "extraction_mode": doc_config.extraction_mode,
                "priority": doc_config.priority,
                "status": "stub_processed"
            }

            # Add placeholder fields
            for field in doc_config.fields:
                record[field] = None

            return [record]

    def _get_document_filename(self, doc_config: DocumentConfig, input_folder: str) -> Optional[str]:
        """Find document file in input folder"""
        input_path = Path(input_folder)

        # Try common patterns
        patterns = [
            f"*{doc_config.name}*.{doc_config.doc_type}",
            f"*{doc_config.ade_template}*.{doc_config.doc_type}",
            f"*.{doc_config.doc_type}"
        ]

        for pattern in patterns:
            matches = list(input_path.glob(pattern))
            if matches:
                return str(matches[0])

        return None

    def _build_schema(self, doc_config: DocumentConfig) -> Dict[str, str]:
        """Build extraction schema from document configuration"""
        # Get template from manifest if available
        templates = self.manifest.get('ade_templates', {})
        template = templates.get(doc_config.ade_template, {})

        if template and 'schema' in template:
            return template['schema']

        # Fallback: build basic schema from fields
        schema = {}
        for field in doc_config.fields:
            # Provide sensible defaults based on field name
            if 'hs_code' in field or 'code' in field:
                schema[field] = "Harmonized System code (10 digits)"
            elif 'description' in field:
                schema[field] = "Product or item description"
            elif 'rate' in field or 'tariff' in field:
                schema[field] = "Rate as percentage or decimal"
            elif 'date' in field:
                schema[field] = "Date in YYYY-MM-DD format"
            elif 'country' in field:
                schema[field] = "Country name or ISO code"
            elif 'price' in field:
                schema[field] = "Price in USD"
            else:
                schema[field] = f"Extract {field}"

        return schema

    def process_category(
        self,
        category: str,
        input_folder: str,
        ade_extractor: Optional[ADEExtractor] = None,
        llm_analyzer: Optional[LLMAnalyzer] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all documents in a category

        Args:
            category: Category name from manifest (e.g., 'tariff_sources')
            input_folder: Folder containing documents
            ade_extractor: ADE extractor instance
            llm_analyzer: LLM analyzer instance

        Returns:
            List of extracted records
        """
        records = []
        documents = [
            doc for doc in self._get_all_documents()
            if self._get_document_category(doc) == category
        ]

        log.info("processing_category", category=category, doc_count=len(documents))

        for doc_config in documents:
            try:
                doc_records = self.process_document(
                    doc_config,
                    input_folder,
                    ade_extractor=ade_extractor,
                    llm_analyzer=llm_analyzer
                )
                records.extend(doc_records)
            except Exception as e:
                log.error(
                    "document_processing_failed",
                    doc=doc_config.name,
                    error=str(e)
                )

        return records

    def _get_document_category(self, doc: DocumentConfig) -> str:
        """Determine which category a document belongs to"""
        # Simple heuristic based on template name
        if 'htsus' in doc.ade_template or 'tariff' in doc.ade_template:
            return 'tariff_sources'
        elif 'commodity' in doc.ade_template or 'gold' in doc.name.lower():
            return 'commodity_prices'
        elif 'cpi' in doc.ade_template or 'inflation' in doc.name.lower():
            return 'inflation_sources'
        elif 'sentiment' in doc.ade_template or 'confidence' in doc.name.lower():
            return 'sentiment_sources'
        elif 'trade' in doc.ade_template or 'ft-900' in doc.name.lower():
            return 'trade_sources'
        elif 'fomc' in doc.ade_template:
            return 'monetary_policy'
        else:
            return 'other'

    def process_all(
        self,
        input_folder: str = "./data/docs",
        output_path: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> str:
        """
        Process all documents according to manifest using parse + extract workflow

        Args:
            input_folder: Folder containing documents to process
            output_path: Override default output path from manifest
            categories: List of categories to process (None = all)

        Returns:
            Path to output parquet file
        """
        if output_path is None:
            output_path = self.manifest.get('metadata', {}).get('output_path', './outputs/ade_llm_data.parquet')

        log.info(
            "processing_all_documents",
            input_folder=input_folder,
            output_path=output_path,
            use_parse_extract=True
        )

        # Initialize extractors
        ade_extractor = ADEExtractor(
            api_key=self.ade_key,
            input_folder=input_folder,
            cache_dir="./cache/ade"
        )

        llm_analyzer = None
        if self.bedrock_model and self.bedrock_region:
            try:
                llm_analyzer = LLMAnalyzer(
                    model_id=self.bedrock_model,
                    region=self.bedrock_region
                )
                log.info("llm_analyzer_initialized", model=self.bedrock_model)
            except Exception as e:
                log.warning("llm_analyzer_init_failed", error=str(e))

        all_records = []

        # Get categories to process
        if categories is None:
            categories = [
                'tariff_sources',
                'commodity_prices',
                'inflation_sources',
                'sentiment_sources',
                'trade_sources',
                'monetary_policy'
            ]

        for category in categories:
            if category in self.manifest:
                records = self.process_category(
                    category,
                    input_folder,
                    ade_extractor=ade_extractor,
                    llm_analyzer=llm_analyzer
                )
                all_records.extend(records)

        # Write to parquet
        if all_records:
            to_parquet(all_records, output_path)
            log.info(
                "processing_complete",
                total_records=len(all_records),
                output_path=output_path
            )
        else:
            log.warning("no_records_extracted")

        return output_path

    def get_manifest_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the manifest"""
        documents = self._get_all_documents()

        summary = {
            'total_documents': len(documents),
            'by_priority': {},
            'by_category': {},
            'llm_enabled': sum(1 for d in documents if d.llm_analysis),
            'extraction_modes': {},
            'doc_types': {}
        }

        for doc in documents:
            # Count by priority
            summary['by_priority'][doc.priority] = summary['by_priority'].get(doc.priority, 0) + 1

            # Count by category
            category = self._get_document_category(doc)
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1

            # Count by extraction mode
            summary['extraction_modes'][doc.extraction_mode] = summary['extraction_modes'].get(doc.extraction_mode, 0) + 1

            # Count by doc type
            summary['doc_types'][doc.doc_type] = summary['doc_types'].get(doc.doc_type, 0) + 1

        return summary


def process_manifest_cli(
    manifest_path: str = "./data/ingestion_manifest.yaml",
    input_folder: str = "./data/docs",
    output_path: str = "./outputs/ade_llm_data.parquet",
    ade_key: Optional[str] = None,
    bedrock_model: Optional[str] = None,
    bedrock_region: Optional[str] = None
):
    """CLI entry point for manifest-based processing"""

    processor = ManifestProcessor(
        manifest_path=manifest_path,
        ade_key=ade_key,
        bedrock_model=bedrock_model,
        bedrock_region=bedrock_region
    )

    # Print summary
    summary = processor.get_manifest_summary()
    log.info("manifest_summary", summary=summary)
    print("\n=== Manifest Summary ===")
    print(f"Total documents: {summary['total_documents']}")
    print(f"\nBy priority:")
    for priority, count in summary['by_priority'].items():
        print(f"  {priority}: {count}")
    print(f"\nBy category:")
    for category, count in summary['by_category'].items():
        print(f"  {category}: {count}")
    print(f"\nLLM-enabled documents: {summary['llm_enabled']}")

    # Process all
    result_path = processor.process_all(
        input_folder=input_folder,
        output_path=output_path
    )

    print(f"\n✓ Processing complete: {result_path}")
    return result_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        manifest_path = sys.argv[1]
    else:
        manifest_path = "./data/ingestion_manifest.yaml"

    process_manifest_cli(manifest_path=manifest_path)
