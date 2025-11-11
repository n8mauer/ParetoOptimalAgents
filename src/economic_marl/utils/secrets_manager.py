from __future__ import annotations
from typing import Optional, Dict, Any
import os
import json
from functools import lru_cache
from ..utils.logging import get_logger

log = get_logger(__name__)


class SecretsManager:
    """Secure secrets management using AWS Secrets Manager with fallback to environment variables.

    This class provides a unified interface for retrieving secrets from:
    1. AWS Secrets Manager (production)
    2. Environment variables (development/fallback)

    Secrets are cached to minimize API calls to AWS.
    """

    def __init__(self, region: str = "us-east-1", use_aws: bool = True):
        """Initialize secrets manager.

        Args:
            region: AWS region for Secrets Manager
            use_aws: If True, attempt to use AWS Secrets Manager. If False, use env vars only.
        """
        self.region = region
        self.use_aws = use_aws
        self._client = None

        if use_aws:
            try:
                import boto3
                self._client = boto3.client("secretsmanager", region_name=region)
                log.info("secrets_manager_initialized", region=region, mode="aws")
            except Exception as e:
                log.warning("secrets_manager_aws_unavailable", error=str(e), fallback="env_vars")
                self._client = None
        else:
            log.info("secrets_manager_initialized", mode="env_vars_only")

    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str, fallback_env_var: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value from AWS Secrets Manager or environment variable.

        Args:
            secret_name: Name of the secret in AWS Secrets Manager
            fallback_env_var: Environment variable name to use as fallback

        Returns:
            Secret value as string, or None if not found

        Example:
            >>> sm = SecretsManager()
            >>> api_key = sm.get_secret("landingai/api_key", fallback_env_var="LANDINGAI_API_KEY")
        """
        # Try AWS Secrets Manager first
        if self._client is not None:
            try:
                response = self._client.get_secret_value(SecretId=secret_name)

                # Secrets can be stored as SecretString or SecretBinary
                if "SecretString" in response:
                    secret_value = response["SecretString"]
                    log.info("secret_retrieved", source="aws_secrets_manager", secret_name=secret_name)
                    return secret_value
                else:
                    # Handle binary secrets if needed
                    log.warning("binary_secret_not_supported", secret_name=secret_name)

            except self._client.exceptions.ResourceNotFoundException:
                log.warning("secret_not_found_in_aws", secret_name=secret_name, fallback="env_var")
            except Exception as e:
                log.error("secret_retrieval_failed", secret_name=secret_name, error=str(e), fallback="env_var")

        # Fallback to environment variable
        if fallback_env_var:
            value = os.getenv(fallback_env_var)
            if value:
                log.info("secret_retrieved", source="environment", env_var=fallback_env_var)
                return value
            else:
                log.warning("secret_not_found", secret_name=secret_name, env_var=fallback_env_var)

        return None

    @lru_cache(maxsize=128)
    def get_secret_json(self, secret_name: str, fallback_env_var: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a secret and parse it as JSON.

        Useful for secrets that contain multiple key-value pairs stored as JSON.

        Args:
            secret_name: Name of the secret in AWS Secrets Manager
            fallback_env_var: Environment variable name to use as fallback

        Returns:
            Parsed JSON as dictionary, or None if not found or invalid JSON

        Example:
            >>> sm = SecretsManager()
            >>> creds = sm.get_secret_json("app/credentials")
            >>> api_key = creds.get("api_key") if creds else None
        """
        secret_value = self.get_secret(secret_name, fallback_env_var)

        if secret_value is None:
            return None

        try:
            return json.loads(secret_value)
        except json.JSONDecodeError as e:
            log.error("secret_json_parse_failed", secret_name=secret_name, error=str(e))
            return None

    def clear_cache(self):
        """Clear the secrets cache. Useful for testing or forcing refresh."""
        self.get_secret.cache_clear()
        self.get_secret_json.cache_clear()
        log.info("secrets_cache_cleared")


# Global singleton instance
_secrets_manager_instance: Optional[SecretsManager] = None


def get_secrets_manager(region: Optional[str] = None, use_aws: Optional[bool] = None) -> SecretsManager:
    """Get or create the global SecretsManager singleton.

    Args:
        region: AWS region (uses default if None)
        use_aws: Whether to use AWS Secrets Manager (uses env var if None)

    Returns:
        SecretsManager instance
    """
    global _secrets_manager_instance

    if _secrets_manager_instance is None:
        if region is None:
            region = os.getenv("AWS_REGION", "us-east-1")
        if use_aws is None:
            use_aws = os.getenv("USE_AWS_SECRETS_MANAGER", "true").lower() in ("true", "1", "yes")

        _secrets_manager_instance = SecretsManager(region=region, use_aws=use_aws)

    return _secrets_manager_instance
