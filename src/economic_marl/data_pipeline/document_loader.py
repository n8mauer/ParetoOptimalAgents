from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import pyarrow as pa
import pyarrow.parquet as pq
from ..utils.logging import get_logger
from .ade_extractor import ADEExtractor
from .llm_analyzer import LLMAnalyzer

log = get_logger(__name__)

def to_parquet(records: List[Dict[str, Any]], path: str) -> None:
    table = pa.Table.from_pylist(records)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)
    log.info("parquet_written", path=path, rows=len(records))

def process_documents(
    ade_key: Optional[str],
    input_folder: str,
    model_id: str,
    region: str,
    output_path: str = "./outputs/ade_llm_data.parquet",
) -> str:
    ade = ADEExtractor(api_key=ade_key, input_folder=input_folder)
    llm = LLMAnalyzer(model_id=model_id, region=region)

    records: List[Dict[str, Any]] = []
    for rec in ade.extract_folder("*.pdf"):
        analysis = llm.assess_tariff_sentiment(rec.text)
        payload = {
            "doc_id": rec.doc_id,
            "country": rec.fields.get("country"),
            "commodity_code": rec.fields.get("commodity_code"),
            "tariff_rate": rec.fields.get("tariff_rate"),
            "effective_date": rec.fields.get("effective_date"),
            "sentiment_score": analysis["sentiment_score"],
            "commodity_impact_pct": analysis["commodity_impact_pct"],
            "gold_response_pct": analysis["gold_response_pct"],
            "reasoning": analysis["reasoning"],
        }
        records.append(payload)
        log.info("doc_processed", doc_id=rec.doc_id)
    to_parquet(records, output_path)
    return output_path
