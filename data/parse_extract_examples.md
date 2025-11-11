# Parse + Extract: Concrete Examples for Your Manifest

## Example 1: HTSUS Chapter 71 (Gold) - Multiple Extractions

### Your Current Manifest Entry:
```yaml
- name: "HTSUS Chapter 71 - Gold & Precious Metals"
  url: "https://hts.usitc.gov/"
  doc_type: "pdf"
  extraction_mode: "table"
  focus: "Headings 7108.* (refined gold)"
  fields: [hs_code, description, rate_of_duty_ad_valorem, special_rate, notes]
  ade_template: "htsus_chapter_table"
```

### With Parse + Extract (Better):

**Step 1: Parse Once**
```python
parsed_ch71 = ade.parse(
    document="./data/docs/Chapter_71_2025HTSRev27.pdf",
    split="page"  # Parse each page separately
)

# Save to cache
save_cache(parsed_ch71, "./cache/ade/htsus_ch71_parsed.json")

# Output:
# {
#   "markdown": "# Chapter 71: Natural or Cultured Pearls...\n\n## Heading 7108...",
#   "chunks": [
#     {"text": "7108.12.00 | Gold powder | 6% | Free", "page": 5, "bbox": {...}},
#     {"text": "7108.13.00 | Other unwrought gold | 6% | Free", "page": 5, "bbox": {...}}
#   ],
#   "pages": 12
# }
```

**Step 2: Extract Multiple Schemas from Same Parse**

**Extraction A: Full tariff table**
```python
all_gold_rates = ade.extract(
    markdown=parsed_ch71['markdown'],
    schema={
        "hs_code": "10-digit Harmonized System code",
        "description": "Product description",
        "rate_of_duty_column1": "Column 1 General rate as percentage",
        "rate_of_duty_column2": "Column 2 rate",
        "special_program_rate": "Special program indicators and rates",
        "unit_of_quantity": "Unit of measurement"
    }
)
# Result: 50+ rows with all gold/precious metal headings
```

**Extraction B: Gold-specific (7108.* only)**
```python
refined_gold_only = ade.extract(
    markdown=parsed_ch71['markdown'],
    schema={
        "hs_code": "Codes starting with 7108 (refined gold)",
        "form": "Powder, unwrought, or semi-manufactured",
        "rate_ad_valorem": "Ad valorem rate",
        "purity_requirement": "Minimum purity if specified"
    }
)
# Result: 8-12 rows with just refined gold (7108.*)
```

**Extraction C: Chapter notes (conditional logic)**
```python
chapter_notes = ade.extract(
    markdown=parsed_ch71['markdown'],
    schema={
        "note_number": "Note number (e.g., Note 1, Note 2)",
        "note_text": "Full text of the note",
        "applicable_headings": "Which HS headings this note applies to",
        "conditional_logic": "Any conditional duty rules mentioned"
    }
)
# Result: Chapter-level notes affecting all headings
```

**Benefits:**
- ✅ Parse PDF once (~2 seconds)
- ✅ Extract 3 different views (~0.5 seconds each)
- ✅ Total: 3.5 seconds vs 6+ seconds (3 full extractions)
- ✅ Can add more extractions later without re-parsing

---

## Example 2: Federal Register Notice (Section 301) - Complex Document

### Your Current Manifest Entry:
```yaml
- name: "Notice of Modification - Section 301 Maritime/Shipbuilding"
  url: "https://www.federalregister.gov/documents/2025/10/16"
  date: "2025-10-16"
  doc_type: "pdf"
  extraction_mode: "table+annex"
  fields: [hs_code, description, additional_duty_301, effective_date, annex_table_number]
  ade_template: "fr_notice_annex"
```

### With Parse + Extract (Better):

**Step 1: Parse**
```python
parsed_fr = ade.parse(
    document_url="https://www.federalregister.gov/documents/2025/10/16",
    split="page"
)

# FR notices are long (30-50 pages)
# Parsing gives clean structure:
# - Summary section
# - Background
# - Annex tables
# - Legal text
```

**Step 2: Multiple Extractions**

**Extraction A: Annex tables (tariff data)**
```python
annex_tables = ade.extract(
    markdown=parsed_fr['markdown'],
    schema={
        "annex_number": "Annex I, Annex II, etc.",
        "hs_code": "10-digit HS code",
        "description": "Product description",
        "additional_duty_301": "Additional duty rate as percentage",
        "effective_date": "Date in YYYY-MM-DD format",
        "sunset_date": "Expiration date if temporary"
    }
)
```

**Extraction B: Policy metadata**
```python
policy_meta = ade.extract(
    markdown=parsed_fr['markdown'],
    schema={
        "fr_citation": "Federal Register citation",
        "docket_number": "USTR docket number",
        "comment_deadline": "Public comment deadline date",
        "affected_industries": "List of industry sectors",
        "estimated_trade_value": "Dollar value of affected imports"
    }
)
```

**Extraction C: Text for LLM analysis**
```python
text_sections = ade.extract(
    markdown=parsed_fr['markdown'],
    schema={
        "executive_summary": "Summary section full text",
        "background": "Background section full text",
        "rationale": "Rationale for modification",
        "stakeholder_comments": "Summary of comments received"
    }
)

# Send to Bedrock for sentiment
sentiment = bedrock_llm.analyze(
    text=text_sections['rationale'],
    grounding=parsed_fr['chunks'],  # Can cite specific paragraphs
    prompt="Assess trade policy sentiment and commodity impacts"
)
```

**Benefits:**
- ✅ Parse 40-page FR notice once (~5 seconds)
- ✅ Extract tariff tables, metadata, AND text (~1.5 seconds total)
- ✅ Grounding for LLM citations
- ✅ Can re-extract if schema needs adjustment

---

## Example 3: FOMC Minutes - LLM Grounding Use Case

### Your Current Manifest Entry:
```yaml
- name: "FOMC Calendar + Minutes Hub (2025)"
  url: "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
  doc_type: "html/pdf"
  extraction_mode: "meeting_dates+documents"
  llm_analysis: true  # ← Uses Bedrock LLM
  llm_prompts:
    - "Extract hawkish/dovish signals regarding inflation and policy path"
```

### With Parse + Extract + LLM Grounding:

**Step 1: Parse**
```python
parsed_fomc = ade.parse(
    document_url="https://www.federalreserve.gov/.../minutes20250917.pdf"
)

# Get chunks with bounding boxes
# {
#   "text": "The Committee decided to maintain the target range...",
#   "page": 3,
#   "bbox": {"x1": 100, "y1": 200, "x2": 500, "y2": 250},
#   "confidence": 0.98
# }
```

**Step 2: Extract Structured Data**
```python
meeting_data = ade.extract(
    markdown=parsed_fomc['markdown'],
    schema={
        "meeting_date": "Meeting date YYYY-MM-DD",
        "policy_decision": "Interest rate decision",
        "vote_split": "Vote count (e.g., 11-1)",
        "dissenting_member": "Name of dissenting member if any",
        "next_meeting_date": "Date of next scheduled meeting"
    }
)
```

**Step 3: Extract Text for LLM with Grounding**
```python
# Extract key sections
sections = ade.extract(
    markdown=parsed_fomc['markdown'],
    schema={
        "statement_text": "Official policy statement",
        "inflation_discussion": "All paragraphs discussing inflation",
        "labor_market_discussion": "All paragraphs discussing employment",
        "policy_outlook": "Forward guidance section"
    }
)

# Send to Bedrock with grounding references
hawkishness = bedrock_llm.invoke({
    "model": "anthropic.claude-3-sonnet",
    "messages": [{
        "role": "user",
        "content": f"""
        Analyze this FOMC statement and score hawkishness (-1 dovish to +1 hawkish).

        Statement: {sections['statement_text']}
        Inflation discussion: {sections['inflation_discussion']}

        Provide:
        1. hawkishness_score: float
        2. inflation_concern: 0-1 scale
        3. evidence: List of specific quotes with page numbers

        Available chunks for grounding: {json.dumps(parsed_fomc['chunks'][:10])}
        """
    }]
})

# Result includes citations:
# {
#   "hawkishness_score": 0.6,
#   "inflation_concern": 0.75,
#   "evidence": [
#     {
#       "quote": "inflation remains elevated and persistent",
#       "page": 3,
#       "chunk_id": 42,
#       "bbox": {...}
#     }
#   ]
# }
```

**Benefits:**
- ✅ LLM can cite specific locations in PDF
- ✅ Bounding boxes for verification/audit
- ✅ Can re-run LLM analysis with different prompts (no re-parse)
- ✅ Structured data + unstructured analysis combined

---

## Example 4: World Bank Pink Sheet - Simple Case

### Your Current Manifest Entry:
```yaml
- name: "Pink Sheet - Gold Spot (Monthly Average)"
  url: "https://www.worldbank.org/en/research/commodity-markets"
  doc_type: "csv/xlsx/pdf"
  extraction_mode: "time_series"
  fields: [date, gold_spot_usd_per_oz, monthly_average, data_source_note]
  ade_template: "commodity_time_series"
```

### With Parse + Extract (Still Beneficial):

**Step 1: Parse**
```python
parsed_pink_sheet = ade.parse(
    document_url="https://www.worldbank.org/.../pink_sheet_2025.xlsx"
)

# Parse works on CSV/XLSX too!
# Converts to clean markdown table
```

**Step 2: Extract Time Series**
```python
gold_prices = ade.extract(
    markdown=parsed_pink_sheet['markdown'],
    schema={
        "date": "Date in YYYY-MM format",
        "gold_spot_usd_per_oz": "Gold price in USD per troy ounce",
        "mom_change": "Month-over-month percentage change",
        "yoy_change": "Year-over-year percentage change"
    }
)
```

**Why still use separate APIs here?**
- ✅ If schema changes (add volatility field), don't re-download
- ✅ Can extract multiple time windows from same parse
- ✅ Consistent with other manifest documents

---

## Cost Comparison: Your 45-Document Manifest

### Scenario: 3 Schema Iterations (Typical Development)

**Single API Approach:**
```
Parse + Extract Combined:
- 45 docs × 3 iterations = 135 API calls
- 135 × $0.10/call = $13.50
```

**Separate API Approach:**
```
Parse: 45 docs × 1 time = 45 calls × $0.10 = $4.50
Extract: 45 docs × 3 iterations = 135 calls × $0.02 = $2.70
Total: $7.20

Savings: $6.30 (47%)
```

### Breakdown by Category:

| Category | Documents | Iterations | Parse Cost | Extract Cost | Single API Cost | Savings |
|----------|-----------|------------|------------|--------------|-----------------|---------|
| Tariff Sources | 18 | 3 | $1.80 | $1.08 | $5.40 | $2.52 |
| Commodity Prices | 8 | 2 | $0.80 | $0.32 | $1.60 | $0.48 |
| Inflation | 5 | 2 | $0.50 | $0.20 | $1.00 | $0.30 |
| Sentiment | 4 | 2 | $0.40 | $0.16 | $0.80 | $0.24 |
| Trade | 3 | 2 | $0.30 | $0.12 | $0.60 | $0.18 |
| Monetary Policy | 2 | 3 | $0.20 | $0.12 | $0.60 | $0.28 |
| Bilateral | 2 | 2 | $0.20 | $0.08 | $0.40 | $0.12 |
| Market Positioning | 1 | 2 | $0.10 | $0.04 | $0.20 | $0.06 |
| **TOTAL** | **45** | **Avg 2.4** | **$4.30** | **$2.12** | **$10.60** | **$4.18** |

---

## Implementation Checklist

### Phase 1: Add Parse Support
```python
# ade_extractor.py
class ADEExtractor:
    def parse_document(self, path, use_cache=True):
        # Implementation
```

### Phase 2: Add Extract Support
```python
    def extract_fields(self, markdown, schema):
        # Implementation
```

### Phase 3: Enable Caching
```bash
mkdir -p ./cache/ade
```

### Phase 4: Update Manifest Processor
```python
# manifest_processor.py
def process_document(self, doc_config):
    parsed = self.ade.parse_document(...)  # Cached
    extracted = self.ade.extract_fields(...)  # Schema-based
```

### Phase 5: Test with Single Document
```python
# Test HTSUS Chapter 71
python -c "
from economic_marl.data_pipeline.ade_extractor import ADEExtractor
ade = ADEExtractor(api_key='...', input_folder='./data/docs')
parsed = ade.parse_document('./data/docs/Chapter_71_2025HTSRev27.pdf')
print('Parsed pages:', len(parsed['chunks']))
"
```

---

## Summary

**For your Economic MARL project, use Parse + Extract separately because:**

1. ✅ **18 tariff documents** need multiple extractions (rates, notes, metadata)
2. ✅ **6 LLM-enriched documents** benefit from grounding (bounding boxes)
3. ✅ **12+ extraction templates** require schema iteration (47% cost savings)
4. ✅ **Caching parsed docs** enables offline development and testing
5. ✅ **Consistent workflow** across all 45 documents in manifest

**Start here:**
1. Implement parse caching for HTSUS Chapter 71 and Chapter 99
2. Test multiple extractions (tariff rates, gold headings, notes)
3. Measure speed/cost improvements
4. Roll out to remaining 43 documents
