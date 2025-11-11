from __future__ import annotations
import typer
from typing import Optional
from .config import SETTINGS
from .utils.logging import get_logger
from .data_pipeline.document_loader import process_documents
from .data_pipeline.manifest_processor import process_manifest_cli
from .training.maddpg_trainer import MADDPGTrainer

app = typer.Typer(help="Economic MARL CLI")
log = get_logger(__name__)

@app.command()
def ingest(
    input_folder: str = typer.Option("./data/docs", help="Folder containing PDFs"),
    output_path: str = typer.Option("./outputs/ade_llm_data.parquet", help="Structured parquet output"),
    manifest: Optional[str] = typer.Option(None, help="Path to YAML ingestion manifest (optional)"),
):
    """Run LandingAI ADE extraction + Bedrock LLM analysis and write parquet.

    Two modes:
    1. Simple mode (no manifest): Process all PDFs in input_folder
    2. Manifest mode (--manifest): Process documents according to YAML configuration
    """
    s = SETTINGS

    # Retrieve API key from secrets manager or environment
    ade_key = s.ade.get_api_key()
    if not ade_key:
        log.warning("ade_api_key_not_found", message="Using stub extractor without API key")

    # Manifest mode
    if manifest:
        typer.echo(f"Using manifest: {manifest}")
        path = process_manifest_cli(
            manifest_path=manifest,
            input_folder=input_folder,
            output_path=output_path,
            ade_key=ade_key,
            bedrock_model=s.bedrock.model_id,
            bedrock_region=s.bedrock.region
        )
    # Simple mode
    else:
        typer.echo("Simple mode: processing all PDFs in input folder")
        path = process_documents(
            ade_key=ade_key,
            input_folder=input_folder,
            model_id=s.bedrock.model_id,
            region=s.bedrock.region,
            output_path=output_path,
        )

    typer.echo(f"Structured data written: {path}")

@app.command()
def train(
    episodes: int = typer.Option(SETTINGS.training.episodes, help="Number of episodes"),
    output_dir: str = typer.Option(SETTINGS.output.output_dir, help="Output directory"),
):
    """Train MADDPG agents in the economic environment and write metrics to parquet."""
    cfg = SETTINGS.training.model_dump()
    trainer = MADDPGTrainer(cfg, output_dir=output_dir)
    metrics_path = trainer.train(
        episodes=episodes,
        max_steps=cfg.get("max_steps_per_episode", 200),
        save_every=SETTINGS.training.save_every,
    )
    typer.echo(f"Training complete → {metrics_path}")

if __name__ == "__main__":
    app()
