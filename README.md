# ParetoOptimal - Multi-Agent Economic Trading System

**LandingAI AI Financial Hackathon Championship**

A production-grade, multi-layer agentic AI system for economic trading signals. Combines **LandingAI Agentic Document Extraction (ADE)**, **Multi-Agent Reinforcement Learning (MADDPG + QMIX)**, and **AWS Bedrock LLM reasoning** to extract real-world economic data and learn Pareto-optimal trading strategies.

## Agentic System Overview

ParetoOptimal integrates **three distinct backend agentic frameworks** working together to process economic documents, learn trading strategies, and generate coordinated investment signals.

### 1. LandingAI Agentic Document Extraction (ADE)

**Purpose**: Autonomous agents that intelligently parse and extract structured data from complex economic documents.

**How It Works**:
![System Screenshot](images/Screenshot%202025-11-10%20121304.png)
- **Parse Agent**: Converts PDFs → Markdown with spatial grounding (bounding boxes, page structure)
- **Extract Agent**: Uses schema-driven extraction to identify and extract specific fields (tariff rates, HS codes, dates)
- **Adaptive Intelligence**: Understands table layouts, multi-column formats, and document relationships

**Documents Processed**: 40 sources across 7 categories
- Tariff schedules (HTSUS chapters, Section 301/232 actions, WTO baselines, and USITC research)
- Commodity price feeders (World Bank, IMF, CME COMEX, USGS, World Gold Council)
- Sentiment proxies (UMich, NY Fed SCE, Conference Board, Fed dollar index)
- Trade balance releases (Census FT‑900, BEA international accounts)
- Monetary-policy texts (FOMC calendars and meeting sets)
- Bilateral policy references (Phase One agreement, USTR WTO compliance)
- Optional market-positioning data (CFTC COT)

**Example Agent Task**:
```
Input: "HTSUS Chapter 99 - Temporary Additional Duties.pdf" (50 pages)
ADE Agent Actions:
  1. Parse complex tariff table structure → Markdown
  2. Identify HS codes, duty rates, effective dates
  3. Extract: {hs_code: "9903.88.01", rate: "25%", date: "2018-07-06"}
Output: Structured tariff records
```

**Code**: [src/economic_marl/data_pipeline/ade_extractor.py](src/economic_marl/data_pipeline/ade_extractor.py)

### 2. Multi-Agent Reinforcement Learning (MADDPG + QMIX)

**Purpose**: Autonomous trading agents that learn coordinated strategies through trial and error.

**Agent Architecture**: 3 specialized agents coordinating via QMIX
- **Gold Agent** - Learns gold futures/physical gold trading strategies
- **Tariff Agent** - Adapts to tariff policy changes and arbitrage opportunities
- **Sentiment Agent** - Trades based on market sentiment and risk signals

**How It Works**:
- Each agent has Actor (decision) + Critic (evaluation) neural networks
- Agents learn optimal policies by interacting with economic environment
- **QMIX** mixing network enables Pareto-optimal coordination
  - Monotonic mixing: Better individual performance → better global outcomes
  - Credit assignment: Agents learn which actions contributed to success
  - Decentralized execution: Agents act independently using local observations

**Training Loop**:
```python
for episode in range(1000):
    # 1. Agents observe environment (seeded with real ADE data)
    states = env.reset()  # Includes real tariff rates, gold prices

    # 2. Each agent autonomously chooses action
    gold_action = gold_agent.act(state)      # "Buy 10 units"
    tariff_action = tariff_agent.act(state)  # "Hedge exposure"
    sentiment_action = sentiment_agent.act(state)  # "Reduce risk"

    # 3. Environment responds with rewards
    next_states, rewards, dones = env.step([gold_action, tariff_action, sentiment_action])

    # 4. Agents learn from experience using QMIX coordination
    for agent in agents:
        agent.update(batch, all_agents, use_qmix=True)
```

**Code**: [src/economic_marl/environment/agents.py](src/economic_marl/environment/agents.py)

### 3. AWS Bedrock LLM Reasoning (Claude Sonnet 4.5)

**Purpose**: Language model agents that provide semantic analysis and reasoning about economic documents.

**Model**: `anthropic.claude-sonnet-4-5-20250929-v1:0`

**Agent Tasks**:
- **Sentiment Analysis**: Analyze FOMC statements for monetary policy stance (hawkish/dovish)
- **Tariff Impact Assessment**: Evaluate economic impact of trade policy changes
- **Document Summarization**: Extract key insights from Federal Register notices
- **Relationship Extraction**: Identify causal relationships in economic data

**Example**:
```python
llm_agent = LLMAnalyzer(model_id="claude-sonnet-4.5")

result = llm_agent.analyze(
    text=fomc_statement_markdown,
    prompt_type="monetary_policy_sentiment"
)

# Output: {
#   "stance": "hawkish",
#   "confidence": 0.87,
#   "key_phrases": ["inflation concerns", "rate hike likely"],
#   "impact_on_gold": "negative",
#   "reasoning": "Fed signals tightening bias..."
# }
```

This enriches ADE-extracted data with semantic understanding and economic reasoning.

**Code**: [src/economic_marl/data_pipeline/llm_analyzer.py](src/economic_marl/data_pipeline/llm_analyzer.py)

### How All Three Agentic Systems Work Together

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **STAGE 1:<br/>Document Processing<br/>(ADE Agents)** | 40 PDF documents<br/>• Tariffs<br/>• Commodities<br/>• Sentiment | **ADE Agents Parse + Extract:**<br/>• Parse documents → Markdown + structure<br/>• Extract fields → structured records<br/>• Optional: LLM agents enrich with sentiment analysis | `./outputs/ade_llm_data.parquet`<br/>(structured data) |
| **STAGE 2:<br/>Environment Seeding<br/>(Data Seeder)** | Parquet data from ADE agents | **Data Seeder transforms:**<br/>• Tariff rates → environment parameters<br/>• Commodity prices → state observations<br/>• Sentiment scores → market conditions | Seeded TradingEnvironment<br/>(real-world data ready for RL) |
| **STAGE 3:<br/>Multi-Agent Training<br/>(MARL Agents)** | Real-world seeded environment | **MARL Agents Learn:**<br/>• Gold Agent learns gold trading strategies<br/>• Tariff Agent learns tariff arbitrage<br/>• Sentiment Agent learns sentiment-based trades<br/>• QMIX coordinates for Pareto optimality | Trained agent policies<br/>(actor neural networks in .pt files) |
| **STAGE 4:<br/>Trading Signals<br/>(Streamlit Dashboard)** | Trained agent policies<br/>+<br/>Current market state | **Agents Generate Signals:**<br/>• Each agent evaluates current conditions<br/>• Generates buy/sell/hold recommendations<br/>• Dashboard displays coordinated signals<br/>• Timeline-specific strategies (short/medium/long-term) | Investment recommendations<br/>with position strategies |

**Agentic Properties**:
- **Autonomy**: All systems make independent decisions
- **Goal-Oriented**: Clear objectives (extract data, maximize profit, analyze sentiment)
- **Learning**: ADE adapts to documents, MARL learns from experience, LLM reasons about context
- **Coordination**: QMIX enables Pareto-optimal multi-agent coordination
- **Adaptability**: Systems adjust to new data, changing markets, different documents
  
### Real-World Integration Example

**Scenario**: Processing Section 301 tariff changes and generating trading signals

1. **ADE Agent** extracts data:
   ```
   Input: "Notice of Modification - Section 301 Maritime Shipbuilding.pdf"

   Agent Actions:
   - Parse 30-page Federal Register notice
   - Extract: [{hs_code: "8901.10.00", rate: "25%", date: "2025-10-16"}]

   Output: Structured tariff change records
   ```

2. **LLM Agent** enriches with sentiment:
   ```python
   llm_analysis = {
       "sentiment_score": -0.6,  # Negative for trade relations
       "commodity_impact_pct": 0.08,  # 8% commodity price increase
       "gold_response_pct": 0.05  # 5% gold safe-haven demand
   }
   ```

3. **Data Seeder** loads into environment:
   ```python
   env.update_tariff_schedule(tariff_changes)
   env.update_sentiment(llm_analysis)
   # Environment now reflects real-world Section 301 changes
   ```

4. **MARL Agents** learn and decide:
   ```python
   # Tariff Agent observes new 25% rate
   tariff_action = tariff_agent.act(state)
   # → "Reduce maritime imports exposure"

   # Gold Agent detects safe-haven opportunity
   gold_action = gold_agent.act(state)
   # → "Increase gold long position by 15%"

   # QMIX coordinates all agents for optimal global outcome
   coordinated_reward = mixer([q_gold, q_tariff, q_sentiment], global_state)
   ```

5. **Dashboard** displays coordinated signal:
   ```
   Medium-Term Strategy (3-12 months):
   - Position: LONG gold, SHORT maritime shipping
   - Gold Signal: BUY (+15% allocation)
   - Tariff Strategy: Reduce import-dependent sectors
   - Confidence: 87%
   ```

## Quick start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate  # Unix/Mac
# OR
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your AWS credentials and LandingAI API key
```

> **Credentials**: Configure AWS and LandingAI credentials securely using AWS Secrets Manager (recommended) or environment variables. See [AWS_STATUS.md](AWS_STATUS.md) and `.env.example`.

### 2. Data Preprocessing (Optional - Enhanced Training)

Extract real-world economic data to seed the training environment:

```bash
# Simple mode: Process PDFs in a folder
python -m economic_marl.cli ingest --input-folder ./data/docs

# Manifest mode (Recommended): Process 45+ structured data sources
python -m economic_marl.cli ingest --manifest ./data/ingestion_manifest.yaml
```

**Data sources processed by manifest:**
- Tariff rates (HTSUS, Section 301, Section 232, WTO)
- Commodity prices (World Bank, IMF, CME COMEX Gold)
- Inflation (BLS CPI/PPI, BEA PCE)
- Sentiment (UMich, NY Fed, Conference Board)
- Trade balance (Census FT-900)
- Monetary policy (FOMC statements with LLM scoring)

### 3. Training

```bash
# Train with real-world data seeding (if preprocessed)
python -m economic_marl.cli train --episodes 200 --output-dir ./outputs

# OR train with synthetic data only
python -m economic_marl.cli train --episodes 200
```

### 4. Dashboard

```bash
# Launch interactive trading signals dashboard
streamlit run dashboard/app.py
```

## Structure

```
src/economic_marl/
    data_pipeline/     # ADE + Bedrock ingestion
    environment/       # Gymnasium env + agents + QMIX
    training/          # MADDPG trainer + evolutionary meta-layer
    utils/             # structured logging
dashboard/
schema/
pipeline/
terraform/
```

## QMIX for Pareto Coordination

The system uses **QMIX (Q-Mixing)** to enable Pareto-optimal coordination among agents:

- **Monotonic Value Mixing**: Enforces that improving individual agent Q-values improves global objectives
- **Credit Assignment**: Learns to attribute global rewards to individual agent actions
- **Centralized Training, Decentralized Execution**: Agents coordinate during training but act independently

QMIX is **enabled by default**. To disable:
```bash
TRAINING__USE_QMIX=false python -m economic_marl.cli train
```

## Real-World Data Seeding with LandingAI ADE

The environment can be seeded with **real-world economic data** extracted from documents using LandingAI ADE:

### Step 1: Extract Data from Documents

**Two modes available:**

#### Simple Mode (All PDFs in folder)
```bash
python -m economic_marl.cli ingest --input-folder ./data/docs --output-path ./outputs/ade_llm_data.parquet
```

#### Manifest Mode (Recommended - Structured data sources)
```bash
python -m economic_marl.cli ingest \
    --manifest ./data/ingestion_manifest.yaml \
    --input-folder ./data/docs \
    --output-path ./outputs/ade_llm_data.parquet
```

The **manifest** (`data/ingestion_manifest.yaml`) defines comprehensive data sources:
- **Tariff rates**: HTSUS schedules, Section 301 (China), Section 232 (Steel/Aluminum), WTO baselines
- **Commodity prices**: World Bank, IMF, CME COMEX Gold futures, USGS, World Gold Council
- **Inflation**: BLS CPI/PPI, BEA PCE Price Index
- **Sentiment/Fiat demand**: UMich, NY Fed SCE, Conference Board, Fed USD Index
- **Trade balance**: Census FT-900, BEA Current Account
- **Monetary policy**: FOMC statements/minutes with LLM hawkishness scoring
- **Policy context**: Phase One Agreement, USTR reports

This command:
- Uses **LandingAI ADE** to extract structured data from documents
- Uses **AWS Bedrock LLM** to analyze sentiment and economic impacts
- Outputs unified structured data to `./outputs/ade_llm_data.parquet`

### Step 2: Train with Real Data
The training environment automatically uses the extracted data if available:

```bash
# Training will automatically seed initial states from ./outputs/ade_llm_data.parquet
python -m economic_marl.cli train --episodes 200 --output-dir ./outputs
```

**What gets seeded from real data:**
- **Tariff rates**: Sampled from actual extracted tariff schedules
- **Commodity prices**: Inferred from commodity impact analysis
- **Gold prices**: Inferred from gold response analysis
- **Inflation rates**: Derived from tariff relationships
- **Fiat demand**: Calculated from sentiment scores
- **Trade balance**: Modeled from tariff impacts

**Fallback behavior**: If `./outputs/ade_llm_data.parquet` doesn't exist, the environment automatically falls back to synthetic defaults.

**To disable real data seeding:**
```python
from economic_marl.environment.economic_env import EconomicEnv, EconomicEnvConfig

cfg = EconomicEnvConfig(use_real_data=False)
env = EconomicEnv(cfg)
```

### Data Categories

#### 1. Tariff Rates (18 sources) - **CRITICAL**
**Purpose**: Ground truth for trade policy regime

- **HTSUS Schedules**
  - Chapter 99 (Temporary/Additional Duties) - table extraction
  - Chapter 71 (Gold, Precious Metals) - Headings 7108.*
  - General Notes (GN3, conditional duty logic)

- **Section 301 (China)**
  - Federal Register notices (2018-2025 modifications)
  - Product exclusion extensions with validity windows
  - USTR Four-Year Review portal linkage

- **Section 232 (Steel/Aluminum)**
  - Proclamations with country exclusions
  - BIS inclusion procedures

- **WTO Baselines**
  - Applied & bound tariffs (US/China)
  - Tariff profiles (simple/trade-weighted averages)

**Extraction**: Table detection, HS code normalization, effective date parsing
**LLM**: Sentiment scoring on policy text
**Output Fields**: `hs_code`, `tariff_rate`, `additional_duty_301`, `additional_duty_232`, `effective_date`, `sentiment_score`

#### 2. Commodity Prices (7 sources) - **CRITICAL**
**Purpose**: Gold price dynamics for reward shaping

- **Monthly Series**
  - World Bank Pink Sheet (1960s-present, CSV/XLSX)
  - IMF Primary Commodity Prices (interactive downloads)

- **Daily Microstructure**
  - CME COMEX Gold (GC) settlements
  - Contract specifications (tick size, settlement methodology)

- **Supply/Demand Context**
  - USGS Mineral Commodity Summaries (production, reserves)
  - World Gold Council Demand Trends (quarterly ETF flows, CB purchases)

**Extraction**: Time series alignment, settlement data normalization
**LLM**: Market commentary sentiment
**Output Fields**: `date`, `gold_spot_usd_per_oz`, `settlement_price`, `volume`, `open_interest`, `etf_flows_tonnes`

#### 2a. Inflation Rates (5 sources) - **CRITICAL** //Will be used in future iterations
**Purpose**: Inflation regime for policy response

- **BLS CPI** (CUUR0000SA0) - monthly news releases + CSV
- **BLS PPI** (Final Demand) - upstream pass-through
- **BEA PCE** - Fed's preferred inflation gauge

**Extraction**: Table parsing, component weight extraction
**Output Fields**: `date`, `cpi_index`, `mom_change`, `yoy_change`, `ppi_final_demand`, `pce_headline`, `pce_core`

#### 3. Sentiment / Fiat Demand (4 sources) - **HIGH**
**Purpose**: Proxy for USD demand in simulation

- UMich Consumer Sentiment (UMCSENT via FRED)
- NY Fed Survey of Consumer Expectations (inflation expectations)
- Conference Board Consumer Confidence (press releases)
- Fed H.10 Nominal Broad USD Index (DTWEXBGS)

**Extraction**: Index values, expectations sub-components
**LLM**: Press release text sentiment scoring
**Output Fields**: `date`, `sentiment_index`, `inflation_expectations_1yr`, `confidence_index`, `dollar_index_value`

#### 4. Trade Balance (3 sources) - **HIGH**
**Purpose**: Trade deficit/surplus dynamics

- Census/BEA FT-900 (monthly goods/services balance)
- BEA International Transactions (quarterly current account)

**Extraction**: Exhibit table parsing, country-specific balances
**Output Fields**: `date`, `trade_balance_total`, `goods_balance`, `services_balance`, `china_balance`

#### 5. Monetary Policy (2 sources) - **MEDIUM**
**Purpose**: FOMC regime features

- FOMC Calendars + Meeting Minutes (2025)
- Statement text with meeting dates

**Extraction**: Meeting date linkage, text extraction
**LLM**: Hawkishness scoring (-1 dovish to +1 hawkish), inflation concern analysis
**Output Fields**: `meeting_date`, `statement_text`, `hawkishness_score`, `inflation_concern`, `policy_direction`

#### 6. Bilateral Policy Context (2 sources) - **MEDIUM**
**Purpose**: US-China trade agreement features

- Phase One Agreement text (commitment timelines)
- USTR WTO Compliance Reports (enforcement cues)

**Extraction**: Legal text chunking, commitment extraction
**LLM**: Regime shift narrative analysis

#### 7. Market Positioning (1 source) - **LOW**
**Purpose**: Leveraged money positioning signals

- CFTC Commitments of Traders (Gold/Metals, weekly)

**Extraction**: Disaggregated report parsing
**Output Fields**: `report_date`, `money_manager_long`, `money_manager_short`, `leveraged_money_net`

### Processing Stages

#### Stage 1: Document Download
```yaml
download:
  retry_attempts: 3
  timeout_seconds: 120
  user_agent: "Economic-MARL-Data-Pipeline/1.0"
```

- Fetches PDFs, CSVs, XLS from URLs
- Caches to `./data/docs/` for reuse
- Handles redirects and pagination

#### Stage 2: ADE Extraction
```yaml
extraction:
  api_key_source: "aws_secrets_manager"
  batch_size: 10
  parallel_processing: true
  max_workers: 4
```

- LandingAI ADE SDK table detection
- Column mapping via template (`htsus_chapter_table`, `commodity_time_series`, etc.)
- Post-processing: HS code normalization, date parsing, duty rate conversion

#### Stage 3: LLM Enrichment (Bedrock)
```yaml
llm_enrichment:
  credentials_source: "iam"
  batch_size: 5
  rate_limit_rpm: 60
```

- Applies to 6 document types marked `llm_analysis: true`
- **Tariff sentiment**: Extracts `sentiment_score`, `commodity_impact_pct`, `gold_response_pct`
- **FOMC hawkishness**: Scores `hawkishness_score`, `inflation_concern`, `policy_direction`
- Uses Anthropic Claude 3 Sonnet via Bedrock

#### Stage 4: Validation
```yaml
validation:
  required_fields:
    tariff: [hs_code, tariff_rate, effective_date]
    commodity: [date, gold_price]
    inflation: [date, cpi_index]
```

- Schema validation per category
- Date format consistency checks
- Outlier detection on numeric fields

#### Stage 5: Output Generation
```yaml
output:
  format: "parquet"
  compression: "snappy"
  partition_by: "document_category"
```

- Unified parquet: `./outputs/ade_llm_data.parquet`
- Schema: All extracted fields + `doc_id`, `url`, `document_category`, `priority`

### Manifest Configuration

The manifest (`data/ingestion_manifest.yaml`) is a declarative YAML configuration defining:

**Metadata**:
```yaml
metadata:
  project: "Economic MARL - Tariff/Commodity/Gold Trading Simulator"
  output_format: "parquet"
  bedrock_model: "anthropic.claude-3-sonnet-20240229-v1:0"
```

**Per-Document Config**:
```yaml
- name: "HTSUS Chapter 71 - Gold & Precious Metals"
  url: "https://hts.usitc.gov/"
  doc_type: "pdf"
  extraction_mode: "table"
  fields: [hs_code, description, rate_of_duty_ad_valorem]
  ade_template: "htsus_chapter_table"
  priority: "critical"
  llm_analysis: false
```

**ADE Templates**:
```yaml
ade_templates:
  htsus_chapter_table:
    extraction_type: "table"
    column_mapping:
      - {source: "Heading/Subheading", target: "hs_code"}
      - {source: "Rates of Duty 1-General", target: "column1_general_rate"}
    post_processing: ["normalize_hs_codes", "parse_duty_rates"]
```

**LLM Prompts**:
```yaml
llm_analysis:
  prompts:
    tariff_sentiment:
      system: "You are an economist analyzing trade policy documents."
      user: |
        Extract JSON with:
        - sentiment_score: -1 (negative) to +1 (positive)
        - commodity_impact_pct: Expected price impact
        - gold_response_pct: Expected gold response
        - reasoning: Brief explanation
```

### Usage Examples

#### View Manifest Summary
```bash
python -m economic_marl.data_pipeline.manifest_processor ./data/ingestion_manifest.yaml

# Output:
# === Manifest Summary ===
# Total documents: 45
# By priority: critical=15, high=12, medium=10, low=8
# LLM-enabled: 6
```

#### Process All Sources
```bash
python -m economic_marl.cli ingest --manifest ./data/ingestion_manifest.yaml
```

#### Process Specific Categories
```python
from economic_marl.data_pipeline.manifest_processor import ManifestProcessor

processor = ManifestProcessor("./data/ingestion_manifest.yaml")
processor.process_all(categories=['tariff_sources', 'commodity_prices'])
```

#### Custom Bedrock Model
```bash
# Override in .env
BEDROCK__MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

python -m economic_marl.cli ingest --manifest ./data/ingestion_manifest.yaml
```

### Integration with Training

The `DataSeeder` class automatically loads `./outputs/ade_llm_data.parquet` during environment initialization:

```python
# src/economic_marl/environment/economic_env.py
class EconomicEnv(gym.Env):
    def __init__(self, cfg: Optional[EconomicEnvConfig] = None):
        if cfg.use_real_data:
            self.data_seeder = DataSeeder(ade_data_path=cfg.ade_data_path)
            # Logs: "data_seeder_enabled, using_real_ade_data"

    def reset(self, ...):
        if self.data_seeder and self.data_seeder.has_real_data():
            initial_state = self.data_seeder.get_initial_state(self.np_random)
            # Samples from real tariff_rates, commodity_impacts, etc.
```

**Seeding logic** ([data_seeder.py:127-170](src/economic_marl/environment/data_seeder.py:127-170)):
- `tariff_rate`: Direct sample from ADE-extracted rates
- `commodity_price`: Base price × (1 + `commodity_impact_pct` from LLM)
- `gold_price`: Base price × (1 + `gold_response_pct` from LLM)
- `inflation`: Derived from tariff_rate × 0.3 + noise
- `fiat_demand`: 1.0 + (`sentiment_score` × 0.2)
- `trade_balance`: Normal(-tariff_rate × 2.0, 0.5)

### Troubleshooting

**No AWS credentials**:
```
NoCredentialsError: Unable to locate credentials
```
→ Add to `.env`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`

**LandingAI key not found**:
```
Warning: ade_api_key_not_found, using stub extractor
```
→ Expected if testing without ADE. Add to `.env`: `ADE_API_KEY=your_key` or store in Secrets Manager.

**Bedrock rate limits**:
```
ThrottlingException: Rate exceeded
```
→ Reduce `llm_enrichment.batch_size` or `rate_limit_rpm` in manifest.

**Invalid YAML**:
```
yaml.scanner.ScannerError: mapping values are not allowed
```
→ Validate syntax at [yamllint.com](http://www.yamllint.com/)

### Documentation

- **Data Seeder**: [src/economic_marl/environment/data_seeder.py](src/economic_marl/environment/data_seeder.py)

## Required Credentials

**ADE and Bedrock are REQUIRED** for document processing and LLM analysis. The system will raise errors on startup if credentials are not configured.

### LandingAI ADE (Required)
```bash
# In .env file:
ADE_API_KEY=your_landingai_api_key_here
```

Get your API key: https://app.landing.ai/

### AWS Bedrock (Required)
```bash
# In .env file:
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
BEDROCK__MODEL_ID=anthropic.claude-sonnet-4-5-20250929-v1:0
```

Or configure via: `aws configure`

**Required IAM Permissions**:
- `bedrock:InvokeModel`
- `bedrock:ListFoundationModels`

## Error Messages

If you see these errors, credentials are missing:

## Notes

- ADE and Bedrock are **REQUIRED** at runtime; the system will fail fast with clear error messages if not configured.
- QMIX mixing network enforces Pareto coordination between agents (can be disabled in config).
