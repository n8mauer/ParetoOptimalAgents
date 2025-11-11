from __future__ import annotations
import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file explicitly (override system environment variables)
load_dotenv(override=True)


class ADEConfig(BaseModel):
    """Configuration for LandingAI ADE (Agentic Document Extraction).

    API key can be sourced from:
    1. AWS Secrets Manager (recommended for production)
    2. Environment variable ADE_API_KEY (fallback)
    """
    api_key: Optional[str] = Field(default=None)
    api_key_secret_name: str = Field(default="landingai/api_key")
    use_secrets_manager: bool = Field(default=True)
    input_folder: str = Field(default="./data/docs")
    s3_bucket: Optional[str] = None
    s3_prefix: str = "landingai/processed/"

    def get_api_key(self) -> Optional[str]:
        """Retrieve API key from secrets manager or environment variable.

        Returns:
            API key string or None if not found
        """
        # If api_key is explicitly set, use it
        if self.api_key:
            return self.api_key

        # Otherwise, use secrets manager
        if self.use_secrets_manager:
            try:
                from economic_marl.utils.secrets_manager import get_secrets_manager
                sm = get_secrets_manager()
                return sm.get_secret(
                    secret_name=self.api_key_secret_name,
                    fallback_env_var="ADE_API_KEY"
                )
            except Exception as e:
                # If secrets manager fails, fall back to environment variable
                import logging
                logging.warning(f"Secrets manager failed, using env var: {e}")
                return os.getenv("ADE_API_KEY")

        # Fallback to environment variable only
        return os.getenv("ADE_API_KEY")

class BedrockConfig(BaseModel):
    """Configuration for AWS Bedrock LLM service."""
    model_id: str = Field(default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    region: str = Field(default="us-east-1")
    max_tokens: int = 4096
    temperature: float = 0.1

class TrainingConfig(BaseModel):
    """Training configuration for MADDPG with optional QMIX coordination."""
    algo: str = "maddpg"  # or "meanfield"
    episodes: int = 1000
    gamma: float = 0.99
    tau: float = 0.01
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 200_000
    max_steps_per_episode: int = 200
    seed: int = 42
    save_every: int = 50

    # QMIX configuration for Pareto coordination
    use_qmix: bool = True
    qmix_embed_dim: int = 32
    lr_mixer: float = 1e-3

class EvolutionConfig(BaseModel):
    population_size: int = 10
    mutation_rate: float = 0.03
    elite_fraction: float = 0.2

class OutputConfig(BaseModel):
    output_dir: str = "./outputs"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "results/checkpoints/"

class AppSettings(BaseSettings):
    log_level: str = "INFO"
    ade: ADEConfig = Field(default_factory=ADEConfig)
    bedrock: BedrockConfig = Field(default_factory=BedrockConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    class Config:
        env_prefix = ""
        env_nested_delimiter = "__"
        # Allow environment variables like:
        # - LOG_LEVEL
        # - ADE__API_KEY_SECRET_NAME
        # - ADE__USE_SECRETS_MANAGER
        # - BEDROCK__MODEL_ID
        # - BEDROCK__REGION
        # - TRAINING__EPISODES
        # etc.

SETTINGS = AppSettings()
