from __future__ import annotations
from typing import Dict, Any
import json
from ..utils.logging import get_logger

log = get_logger(__name__)

class LLMAnalyzer:
    """LLM sentiment & impact analysis using AWS Bedrock (Anthropic Claude).
    REQUIRED - No fallback. System will raise errors if Bedrock is unavailable.
    """

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self.region = region

        # Validate model ID is provided
        if not model_id:
            raise ValueError(
                "BEDROCK__MODEL_ID is required for LLM analysis.\n"
                "Set in .env file: BEDROCK__MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0\n"
                "Recommended: Use cross-region inference profile for Claude Sonnet 4.5"
            )

        # Initialize AWS Bedrock - REQUIRED, no fallback
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError

            # Test AWS credentials first
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()

            # Initialize Bedrock client
            self.bedrock = boto3.client("bedrock-runtime", region_name=region)
            self._has_bedrock = True

            # Note: Skipping list_foundation_models() validation test
            # That method exists on 'bedrock' client (control plane), not 'bedrock-runtime' (data plane)
            # Permission errors will surface on first invoke_model() call with clear error messages

            log.info(
                "bedrock_client_initialized",
                model_id=model_id,
                region=region,
                aws_account=identity['Account']
            )

        except ImportError as e:
            raise RuntimeError(
                "AWS SDK (boto3) is required but not installed.\n"
                "Install it with: pip install boto3\n"
                f"ImportError: {str(e)}"
            ) from e
        except NoCredentialsError as e:
            raise RuntimeError(
                "AWS credentials are required but not found.\n"
                "Configure credentials:\n"
                "  Option 1: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env\n"
                "  Option 2: Run 'aws configure'\n"
                "  Option 3: Use IAM role (if running on AWS EC2/Lambda)\n"
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize AWS Bedrock client: {str(e)}\n"
                "Check your AWS region ({region}) and network connection.\n"
                "Verify Bedrock is available in your region."
            ) from e

    def assess_tariff_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze tariff sentiment using AWS Bedrock - REQUIRED, no fallback."""
        if not self._has_bedrock:
            raise RuntimeError(
                "AWS Bedrock is not available. Cannot proceed with LLM analysis.\n"
                "This should never happen if __init__ validation passed."
            )

        system = (
            "You are an economist. Extract JSON with fields: "
            "sentiment_score (-1..1), commodity_impact_pct, gold_response_pct, reasoning."
        )
        prompt = f"""
        SYSTEM: {system}
        TEXT: {text}
        Respond ONLY in JSON with keys: sentiment_score, commodity_impact_pct, gold_response_pct, reasoning.
        """

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            })
            resp = self.bedrock.invoke_model(modelId=self.model_id, body=body)
            payload = json.loads(resp["body"].read().decode("utf-8"))

            # Debug logging - see what Claude actually returns
            log.info("bedrock_raw_payload", payload=payload)

            content = payload.get("content", [])
            if not content or not isinstance(content[0], dict) or "text" not in content[0]:
                log.error("bedrock_unexpected_format", payload=payload)
                raise ValueError("Unexpected Bedrock response format - missing content[0]['text']")

            response_text = content[0]["text"]
            log.info("bedrock_response_text", text=response_text[:500])  # Log first 500 chars

            # Try to parse JSON from response
            # Claude Sonnet 4.5 may wrap JSON in markdown or add extra text
            parsed = None
            try:
                # First try: Direct JSON parse
                parsed = json.loads(response_text)
            except json.JSONDecodeError as parse_err:
                # Second try: Extract JSON from text using regex
                import re
                log.warning("direct_json_parse_failed", error=str(parse_err), trying="regex_extraction")

                # Look for JSON object in text (handles markdown code blocks, extra text, etc.)
                json_match = re.search(r'\{[^{}]*"sentiment_score"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    log.info("json_extracted_via_regex", parsed=parsed)
                else:
                    log.error("json_extraction_failed", response_text=response_text)
                    raise ValueError(
                        f"Could not extract JSON from Claude response.\n"
                        f"Response text: {response_text[:200]}\n"
                        f"Expected JSON with sentiment_score, commodity_impact_pct, gold_response_pct, reasoning"
                    ) from parse_err

            if not parsed:
                parsed = {}

            log.info("bedrock_parsed", parsed=parsed)
            return {
                "sentiment_score": float(parsed.get("sentiment_score", 0.0)),
                "commodity_impact_pct": float(parsed.get("commodity_impact_pct", 0.0)),
                "gold_response_pct": float(parsed.get("gold_response_pct", 0.0)),
                "reasoning": parsed.get("reasoning", ""),
            }
        except Exception as e:
            log.error("bedrock_invoke_failed", error=str(e), model_id=self.model_id)
            raise RuntimeError(
                f"Failed to invoke AWS Bedrock for sentiment analysis:\n{str(e)}\n"
                "Check your Bedrock permissions and model ID.\n"
                f"Model: {self.model_id}\n"
                "Verify the model is available in your region."
            ) from e

    def analyze(self, text: str, prompt_type: str = "tariff_sentiment") -> Dict[str, Any]:
        """
        Generic analysis method for document enrichment

        This is called by manifest_processor.py during document processing.

        Args:
            text: Document text/markdown to analyze (typically first 4000 chars)
            prompt_type: Type of analysis to perform
                - "tariff_sentiment": Tariff impact and sentiment analysis
                - "monetary_policy": FOMC hawkishness/dovishness analysis
                - "trade_policy": Trade policy regime shift analysis

        Returns:
            Dictionary with analysis results:
            {
                "sentiment_score": float (-1 to 1),
                "commodity_impact_pct": float (0 to 1),
                "gold_response_pct": float (0 to 1),
                "reasoning": str
            }
        """
        # For now, all prompt types use the same underlying analysis
        # Future enhancement: Add specialized prompts for different document types

        if prompt_type in ["tariff_sentiment", "monetary_policy", "trade_policy"]:
            return self.assess_tariff_sentiment(text)
        else:
            log.warning(
                "unknown_prompt_type",
                prompt_type=prompt_type,
                fallback="tariff_sentiment"
            )
            return self.assess_tariff_sentiment(text)
