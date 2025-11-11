"""
Data seeder to initialize economic environment with real-world data from LandingAI ADE
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RealWorldData:
    """Real-world economic data extracted from documents via ADE"""
    tariff_rates: np.ndarray
    commodity_codes: list
    countries: list
    effective_dates: list
    sentiment_scores: np.ndarray
    commodity_impacts: np.ndarray
    gold_responses: np.ndarray

    # Computed aggregates
    avg_tariff_rate: float
    avg_commodity_impact: float
    avg_gold_response: float
    tariff_volatility: float


class DataSeeder:
    """Seeds economic environment with real data from LandingAI ADE extraction"""

    def __init__(self, ade_data_path: Optional[str] = None):
        """
        Initialize data seeder

        Args:
            ade_data_path: Path to ADE extracted data parquet file
                          Defaults to ./outputs/ade_llm_data.parquet
        """
        self.ade_data_path = ade_data_path or "./outputs/ade_llm_data.parquet"
        self.data: Optional[RealWorldData] = None
        self._load_data()

    def _load_data(self):
        """Load and process ADE extracted data"""
        if not os.path.exists(self.ade_data_path):
            log.warning(
                "ade_data_not_found",
                path=self.ade_data_path,
                message="Will use synthetic defaults"
            )
            self.data = None
            return

        try:
            df = pd.read_parquet(self.ade_data_path)
            log.info("ade_data_loaded", path=self.ade_data_path, rows=len(df))

            # Extract fields
            tariff_rates = df['tariff_rate'].fillna(0.05).values
            sentiment_scores = df['sentiment_score'].fillna(0.0).values
            commodity_impacts = df['commodity_impact_pct'].fillna(0.0).values / 100.0
            gold_responses = df['gold_response_pct'].fillna(0.0).values / 100.0

            # Compute aggregates
            avg_tariff = float(np.mean(tariff_rates))
            avg_commodity_impact = float(np.mean(np.abs(commodity_impacts)))
            avg_gold_response = float(np.mean(np.abs(gold_responses)))
            tariff_volatility = float(np.std(tariff_rates))

            self.data = RealWorldData(
                tariff_rates=tariff_rates,
                commodity_codes=df['commodity_code'].fillna("").tolist(),
                countries=df['country'].fillna("").tolist(),
                effective_dates=df['effective_date'].fillna("").tolist(),
                sentiment_scores=sentiment_scores,
                commodity_impacts=commodity_impacts,
                gold_responses=gold_responses,
                avg_tariff_rate=avg_tariff,
                avg_commodity_impact=avg_commodity_impact,
                avg_gold_response=avg_gold_response,
                tariff_volatility=tariff_volatility
            )

            log.info(
                "data_seeder_initialized",
                avg_tariff=avg_tariff,
                tariff_volatility=tariff_volatility,
                avg_commodity_impact=avg_commodity_impact,
                avg_gold_response=avg_gold_response,
                num_records=len(tariff_rates)
            )

        except Exception as e:
            log.error("ade_data_load_failed", error=str(e), path=self.ade_data_path)
            self.data = None

    def get_initial_state(self, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """
        Get initial state values seeded from real data

        Args:
            rng: Random number generator for sampling

        Returns:
            Dictionary with initial state values
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.data is None:
            # Fallback to synthetic defaults
            log.debug("using_synthetic_defaults")
            return {
                'tariff_rate': float(rng.uniform(0.0, 0.15)),
                'commodity_price': float(rng.normal(100.0, 5.0)),
                'gold_price': float(rng.normal(1900.0, 30.0)),
                'inflation': float(rng.uniform(0.0, 0.04)),
                'fiat_demand': 1.0,
                'trade_balance': float(rng.normal(0.0, 0.5))
            }

        # Sample from real data distribution
        sample_idx = rng.integers(0, len(self.data.tariff_rates))

        # Use real tariff rate as base
        tariff_rate = float(self.data.tariff_rates[sample_idx])

        # Infer commodity price from impact data
        # Higher positive impact = expect price increase
        commodity_impact = self.data.commodity_impacts[sample_idx]
        base_commodity_price = 100.0
        commodity_price = float(base_commodity_price * (1.0 + commodity_impact))

        # Infer gold price from response data
        gold_response = self.data.gold_responses[sample_idx]
        base_gold_price = 1900.0
        gold_price = float(base_gold_price * (1.0 + gold_response))

        # Infer inflation from tariff rate (higher tariffs -> higher inflation)
        inflation = float(np.clip(tariff_rate * 0.3 + rng.normal(0.02, 0.01), 0.0, 0.10))

        # Infer fiat demand from sentiment
        sentiment = self.data.sentiment_scores[sample_idx]
        fiat_demand = float(np.clip(1.0 + sentiment * 0.2, 0.5, 1.5))

        # Trade balance affected by tariffs (simplified)
        trade_balance = float(rng.normal(-tariff_rate * 2.0, 0.5))

        log.debug(
            "seeded_initial_state",
            tariff_rate=tariff_rate,
            commodity_price=commodity_price,
            gold_price=gold_price,
            inflation=inflation,
            sample_idx=sample_idx
        )

        return {
            'tariff_rate': tariff_rate,
            'commodity_price': commodity_price,
            'gold_price': gold_price,
            'inflation': inflation,
            'fiat_demand': fiat_demand,
            'trade_balance': trade_balance
        }

    def get_calibrated_sensitivities(self) -> Dict[str, float]:
        """
        Get calibrated sensitivity parameters from real data

        Returns:
            Dictionary with calibrated beta parameters
        """
        if self.data is None:
            log.debug("using_default_sensitivities")
            return {
                'tariff_sensitivity': 0.6,
                'commodity_pass_through': 0.4,
                'gold_hedge_beta': 0.35,
                'inflation_beta': 0.25,
                'fiat_substitution_beta': 0.30
            }

        # Calibrate from real data
        # Use average impacts as proxies for sensitivities
        tariff_sensitivity = float(np.clip(self.data.tariff_volatility * 10.0, 0.3, 1.0))
        commodity_pass_through = float(np.clip(self.data.avg_commodity_impact * 2.0, 0.2, 0.8))
        gold_hedge_beta = float(np.clip(self.data.avg_gold_response * 2.0, 0.2, 0.6))

        # Inflation and fiat betas derived from relationships
        inflation_beta = float(np.clip(commodity_pass_through * 0.6, 0.15, 0.40))
        fiat_substitution_beta = float(np.clip(gold_hedge_beta * 0.8, 0.2, 0.5))

        log.info(
            "calibrated_sensitivities",
            tariff_sensitivity=tariff_sensitivity,
            commodity_pass_through=commodity_pass_through,
            gold_hedge_beta=gold_hedge_beta,
            inflation_beta=inflation_beta,
            fiat_substitution_beta=fiat_substitution_beta
        )

        return {
            'tariff_sensitivity': tariff_sensitivity,
            'commodity_pass_through': commodity_pass_through,
            'gold_hedge_beta': gold_hedge_beta,
            'inflation_beta': inflation_beta,
            'fiat_substitution_beta': fiat_substitution_beta
        }

    def has_real_data(self) -> bool:
        """Check if real data is available"""
        return self.data is not None

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data"""
        if self.data is None:
            return {'status': 'no_data', 'using': 'synthetic_defaults'}

        return {
            'status': 'loaded',
            'using': 'real_world_data',
            'num_records': len(self.data.tariff_rates),
            'avg_tariff_rate': self.data.avg_tariff_rate,
            'tariff_range': [float(self.data.tariff_rates.min()), float(self.data.tariff_rates.max())],
            'avg_commodity_impact': self.data.avg_commodity_impact,
            'avg_gold_response': self.data.avg_gold_response,
            'tariff_volatility': self.data.tariff_volatility,
            'countries': list(set(self.data.countries)),
            'commodity_codes': list(set(self.data.commodity_codes))
        }