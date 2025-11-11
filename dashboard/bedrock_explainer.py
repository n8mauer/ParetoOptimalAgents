"""
AWS Bedrock LLM Integration for Multi-Agent Explainability
Generates natural language explanations for QMIX agent decisions

OPTIONAL: Requires AWS credentials and boto3
"""

import json
from typing import Dict, List, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False


class BedrockExplainer:
    """
    Generate natural language explanations using AWS Bedrock
    Supports Claude 3, Titan, and other foundation models
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region: str = "us-east-1",
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        """
        Initialize Bedrock client

        Args:
            model_id: Bedrock model ID (default: Claude 3 Sonnet)
            region: AWS region
            max_tokens: Maximum response length
            temperature: Creativity (0.0-1.0)
        """
        if not BEDROCK_AVAILABLE:
            raise ImportError("boto3 not installed. Run: pip install boto3")

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")

    def explain_signal(
        self,
        signal: str,
        asset: str,
        agent_data: List[Dict[str, Any]],
        predictions: Any,
        position_type: str
    ) -> str:
        """
        Generate explanation for why a trading signal was generated

        Args:
            signal: BUY/SELL/SHORT/COVER/HOLD
            asset: Gold/Aluminum
            agent_data: List of agent contributions per step
            predictions: DataFrame of predictions
            position_type: LONG/SHORT

        Returns:
            Natural language explanation string
        """

        # Build context for LLM
        context = self._build_context(signal, asset, agent_data, predictions, position_type)

        # Create prompt
        prompt = f"""You are an expert financial analyst explaining automated trading recommendations.

A multi-agent reinforcement learning system has generated the following trading signal:

**Signal:** {signal} {asset}
**Position Type:** {position_type}
**Asset:** {asset}

**Market Context:**
{context}

Please provide a clear, professional explanation of:
1. WHY this signal was generated (key factors)
2. WHAT the agents observed in the market data
3. HOW the multi-agent coordination led to this decision
4. WHAT risks or uncertainties exist

Keep the explanation under 200 words, suitable for an informed retail investor.
"""

        try:
            # Call Bedrock
            response = self._invoke_bedrock(prompt)
            return response

        except Exception as e:
            # Fallback to rule-based explanation
            return self._fallback_explanation(signal, asset, agent_data, predictions)

    def explain_agent_reasoning(
        self,
        agent_name: str,
        agent_state: Dict[str, Any],
        q_value: float,
        signal: str
    ) -> str:
        """
        Generate explanation for a single agent's contribution

        Args:
            agent_name: Name of the agent
            agent_state: Market state as seen by agent
            q_value: Agent's Q-value
            signal: Final trading signal

        Returns:
            Natural language explanation
        """

        prompt = f"""You are explaining how one agent in a multi-agent trading system made its decision.

**Agent:** {agent_name}
**Q-Value:** {q_value:.3f} ({'bullish' if q_value > 0 else 'bearish'})
**Final Signal:** {signal}

**Market State Observed by Agent:**
- Gold Price: ${agent_state['gold_price']:.2f}
- Aluminum Price: ${agent_state['aluminum_price']:.2f}
- Tariff Rate: {agent_state['tariff_rate']*100:.2f}%
- Inflation: {agent_state.get('inflation', 0)*100:.2f}%
- Trade Balance: {agent_state.get('trade_balance', 0):.2f}

Explain in 2-3 sentences:
1. What this agent focuses on
2. What it observed in the data
3. Why it produced this Q-value

Be concise and technical.
"""

        try:
            response = self._invoke_bedrock(prompt)
            return response
        except Exception as e:
            return f"{agent_name} observed market conditions and produced Q={q_value:.3f}. " \
                   f"This {'supports' if q_value > 0 else 'opposes'} the {signal} signal."

    def explain_qmix_coordination(
        self,
        agent_q_values: Dict[str, float],
        q_tot: float,
        signal: str
    ) -> str:
        """
        Explain how QMIX combined individual agent Q-values

        Args:
            agent_q_values: Dict of agent names to Q-values
            q_tot: Combined Q_tot value
            signal: Final trading signal

        Returns:
            Natural language explanation
        """

        agent_summary = "\n".join([f"- {name}: {q:.3f}" for name, q in agent_q_values.items()])

        prompt = f"""Explain how a QMIX mixing network combined individual agent recommendations.

**Individual Agent Q-Values:**
{agent_summary}

**Combined Q_tot:** {q_tot:.3f}
**Final Signal:** {signal}

Explain in 2-3 sentences:
1. How the agents agreed or disagreed
2. How QMIX resolved any conflicts
3. Why this led to the {signal} signal

Use simple language for a non-technical investor.
"""

        try:
            response = self._invoke_bedrock(prompt)
            return response
        except Exception as e:
            agreement = "strong consensus" if max(agent_q_values.values()) - min(agent_q_values.values()) < 0.5 else "mixed opinions"
            return f"The agents showed {agreement}. QMIX combined their recommendations into Q_tot={q_tot:.3f}, " \
                   f"leading to the {signal} signal."

    def _build_context(
        self,
        signal: str,
        asset: str,
        agent_data: List[Dict[str, Any]],
        predictions: Any,
        position_type: str
    ) -> str:
        """Build market context string for LLM prompt"""

        initial = predictions.iloc[0]
        final = predictions.iloc[-1]

        context = f"""
**Price Movement:**
- Gold: ${initial['gold_price']:.2f} -> ${final['gold_price']:.2f} ({((final['gold_price']/initial['gold_price']-1)*100):.2f}%)
- Aluminum: ${initial['aluminum_price']:.2f} -> ${final['aluminum_price']:.2f} ({((final['aluminum_price']/initial['aluminum_price']-1)*100):.2f}%)

**Economic Indicators:**
- Tariff Rate: {initial['tariff_rate']*100:.2f}% -> {final['tariff_rate']*100:.2f}%
- Inflation: {initial.get('inflation', 0)*100:.2f}% -> {final.get('inflation', 0)*100:.2f}%

**Agent Consensus:**
"""

        # Final step agent Q-values
        final_agents = agent_data[-1]['agents']
        for name, data in final_agents.items():
            context += f"- {name}: Q={data['q_value']:.3f}\n"

        q_tot = sum(data['q_value'] for data in final_agents.values())
        context += f"\n**Combined Q_tot:** {q_tot:.3f}"

        return context

    def _invoke_bedrock(self, prompt: str) -> str:
        """
        Call Bedrock API with prompt

        Args:
            prompt: Text prompt

        Returns:
            Model response text
        """

        # Format request based on model
        if "anthropic.claude" in self.model_id:
            # Claude format
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        elif "amazon.titan" in self.model_id:
            # Titan format
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": 0.9
                }
            }
        else:
            raise ValueError(f"Unsupported model: {self.model_id}")

        # Invoke model
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body)
        )

        # Parse response
        response_body = json.loads(response['body'].read())

        if "anthropic.claude" in self.model_id:
            return response_body['content'][0]['text']
        elif "amazon.titan" in self.model_id:
            return response_body['results'][0]['outputText']
        else:
            raise ValueError(f"Cannot parse response from {self.model_id}")

    def _fallback_explanation(
        self,
        signal: str,
        asset: str,
        agent_data: List[Dict[str, Any]],
        predictions: Any
    ) -> str:
        """
        Generate rule-based explanation if Bedrock fails

        Args:
            signal: Trading signal
            asset: Asset name
            agent_data: Agent contributions
            predictions: Prediction data

        Returns:
            Rule-based explanation string
        """

        initial = predictions.iloc[0]
        final = predictions.iloc[-1]

        if asset.lower() == 'gold':
            price_initial = initial['gold_price']
            price_final = final['gold_price']
        else:
            price_initial = initial['aluminum_price']
            price_final = final['aluminum_price']

        price_change = ((price_final / price_initial) - 1) * 100

        final_agents = agent_data[-1]['agents']
        q_tot = sum(data['q_value'] for data in final_agents.values())

        bullish_agents = sum(1 for data in final_agents.values() if data['q_value'] > 0)
        bearish_agents = sum(1 for data in final_agents.values() if data['q_value'] < 0)

        explanation = f"The system generated a {signal} signal for {asset} based on the following analysis:\n\n"

        explanation += f"**Price Forecast:** {asset} is predicted to {'rise' if price_change > 0 else 'fall'} "
        explanation += f"by {abs(price_change):.2f}% (${price_initial:.2f} -> ${price_final:.2f}).\n\n"

        explanation += f"**Agent Consensus:** {bullish_agents} agents are bullish, {bearish_agents} are bearish. "
        explanation += f"The QMIX coordination mechanism combined these views into Q_tot={q_tot:.3f}, "
        explanation += f"which {'supports' if q_tot > 0 else 'contradicts'} the {signal} recommendation.\n\n"

        if signal in ['BUY', 'SHORT']:
            explanation += "**Action:** The model recommends entering a position based on favorable price movement expectations."
        elif signal in ['SELL', 'COVER']:
            explanation += "**Action:** The model recommends exiting to avoid losses or lock in gains."
        else:
            explanation += "**Action:** The model recommends waiting as price movement is insufficient to justify action."

        return explanation


def get_explainer(enabled: bool = True) -> Optional[BedrockExplainer]:
    """
    Factory function to get Bedrock explainer

    Args:
        enabled: Whether to enable Bedrock integration

    Returns:
        BedrockExplainer if available, else None
    """

    if not enabled:
        return None

    if not BEDROCK_AVAILABLE:
        print("WARNING: boto3 not installed. Bedrock explainer unavailable.")
        return None

    try:
        explainer = BedrockExplainer()
        return explainer
    except Exception as e:
        print(f"WARNING: Could not initialize Bedrock: {e}")
        return None
