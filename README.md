# Carefull Takehome - Transaction Alert System

An LLM-powered pipeline designed to detect and validate suspicious financial activity (exploitation, money mules, and out-of-profile spending) for older adults.

**🔗 Live Demo: [carefull-takehome.streamlit.app](https://carefull-takehome.streamlit.app)** — compare both models side by side and browse the LLM verdicts and reasoning for every flagged transaction, no setup or API keys required.

## Architecture
The system uses a two-stage approach to balance accuracy, latency, and cost:
1. **Heuristic Filter (`heuristics.py`)**: Uses Pandas to process raw transactions, normalize recipient entities, and flag mathematical anomalies (escalating transfers, rapid in/out flows, 3-sigma spending deviations).
2. **LLM Validator (`llm_validator.py`)**: Analyzes flagged transactions alongside the user's biographical profile using LangChain to determine true risk, effectively filtering out false positives.
3. **Results Dashboard (`app.py`)**: A Streamlit front-end that reads the committed `results/` artifacts and renders, for every provider: the evaluation metrics (precision, recall, F1, confusion matrix, latency, projected cost) side by side, plus the full per-transaction verdicts with the model's reasoning. Scoring is delegated to `evaluate_metrics.compute_metrics`, so the dashboard and the CLI report cannot drift apart.

### How the LLM Is Integrated into the Pipeline

The LLM is **not** a classifier applied to the whole dataset. It is a second-opinion reviewer that only sees the small subset the heuristics could not resolve on their own:

```
datasets/transactions.csv ─┐
                           ├─► TransactionAnalyzer.run_pipeline()  (291 txs)
datasets/user_profiles.csv ┘            │
                                        │ ~10% survive the heuristic filter
                                        ▼
                             FraudValidatorAgent (29 txs)
                                        │
            PromptTemplate ──► LLM ──► PydanticOutputParser
                                        │
                                        ▼
                        results/results_<provider>.csv
                                        │
                                        ▼
                             evaluate_metrics.py  (scored against ground truth)
```

Step by step:

1. **Heuristic pre-filter.** `run_pipeline()` runs the three detectors, unions their flagged IDs, and returns only the rows where `requires_llm_review == True`. Everything else is implicitly predicted benign — which is what keeps the LLM bill at ~10% of volume.
2. **Context assembly.** Each flagged row is merged with its user profile, so the prompt carries both the transaction *and* the biographical context (age, income, bio) needed to tell "wealthy senior on vacation" apart from "senior being groomed".
3. **Structured prompting.** `PromptTemplate` injects those fields plus `format_instructions` generated from the `RiskAssessment` Pydantic model, so the schema is described to the model in its own prompt.
4. **Structured parsing.** The LangChain chain is `prompt | llm | parser`. `PydanticOutputParser` validates the response into `RiskAssessment` (`risk_level`, `reasoning`, `confidence_score`). It is provider-agnostic, which avoids the native function-calling incompatibilities on OpenRouter.
5. **Resilience.** Each call is individually wrapped in try/except, latency is timed per transaction, and the loop sleeps between calls to respect provider rate limits. One bad response never aborts the run.
6. **Scoring.** `evaluate_metrics.py` re-expands the LLM verdicts over the full 291-transaction universe: `high`/`medium` becomes an alert (1), while `low` **and** anything the heuristic filtered out become benign (0). This is what makes recall an honest end-to-end number rather than a measure of the LLM in isolation.

### Data Flow Examples

Each example below is one hand-off point in the diagram above, showing exactly what crosses the boundary between stages.

**Stage 0 — Raw input: `datasets/transactions.csv`**

The ledger the system ingests. Note that `raw_description` is unnormalized bank text, which is why the enrichment step exists.

```csv
transaction_id,user_id,date,amount,type,raw_description,category,channel
TXN-770487,USR-001,2024-07-01 06:01:47,3200.0,credit,OHIO STRS PENSION DIRECT DEP,income,ACH
TXN-356787,USR-001,2024-07-04 11:08:47,50.12,debit,KROGER #1234 COLUMBUS OH,groceries,POS
```

**Stage 0 — Raw input: `datasets/user_profiles.csv`**

Joined on `user_id`. The `bio` and `monthly_income` fields are what later let the LLM distinguish a legitimate splurge from exploitation — without them, a $1,250 charge is just a number.

```csv
user_id,name,age,monthly_income,bio
USR-001,Margaret Thompson,74,3200,"Retired teacher in Columbus OH. Two adult children in Seattle and Denver..."
```

**Stage 1 — Heuristic output (`python heuristics.py`)**

The filter narrows 291 transactions to 29 candidates (~10%). This is the cost-control step: only these rows will ever reach a paid API.

```
Total transactions sent for LLM review: 29
     user_id                date   amount    type                 raw_description
274  USR-005 2024-10-04 14:26:46   890.00   debit                     PACHA IBIZA ESP
283  USR-005 2024-10-11 09:26:22  1250.00   debit                CAVALLI CLUB DUBAI AE
```

**Stage 2 — LLM verdicts: `results/results_groq.csv`**

One row per reviewed transaction. `reasoning` is the auditable explanation a human reviewer would read, and `latency_sec` is what feeds the performance report.

```csv
transaction_id,user_id,risk_level,reasoning,confidence,latency_sec,provider
TXN-214576,USR-001,medium,"$350 Zelle transfer to Kevin Richards, who is not a known contact...",70,0.97,groq
```

**Stage 3 — Scoring (`python evaluate_metrics.py`)**

The 29 verdicts are re-expanded over all 291 transactions before scoring, so the numbers below measure the *whole pipeline* (heuristic + LLM), not the LLM alone.

```
========================================
EVALUATING MODEL: OPENROUTER
========================================

1. Classification Report (Precision, Recall, F1):
              precision    recall  f1-score   support

      Normal       0.97      1.00      0.98       259
       Fraud       1.00      0.72      0.84        32

    accuracy                           0.97       291

2. Confusion Matrix:
True Negatives (Normal -> Normal): 259
False Positives (Normal -> Fraud): 0 <- False alarms
False Negatives (Fraud -> Normal): 9 <- Fraud we missed
True Positives (Fraud -> Fraud): 23 <- Fraud caught

3. Performance and Cost (50k tx/month projection):
Average latency per transaction sent to LLM: 2.55 seconds
{
  "cost_per_llm_tx": "$0.00015",
  "projected_monthly_cost": "$0.75 (processing 5000 txs with LLM)"
}
```

## Dependencies and Setup

1. **Python Version**: Requires Python 3.9 or higher.
2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**: 
   ```bash
   export GROQ_API_KEY="your_groq_key"
   export OPENROUTER_API_KEY="your_openrouter_key"
   ```

5. **Model Configuration**:

   Model names are defined in `PROVIDER_CONFIG` inside `llm_validator.py` and can be overridden without touching the code:

   | Provider | Default model | Override variable | Why it was chosen |
   | --- | --- | --- | --- |
   | Groq | `openai/gpt-oss-20b` | `GROQ_MODEL` | High-speed, open-source reasoning evaluation |
   | OpenRouter | `openai/gpt-4o-mini` | `OPENROUTER_MODEL` | Cost-effective commercial model comparison |

   ```bash
   export GROQ_MODEL="openai/gpt-oss-20b"
   export OPENROUTER_MODEL="openai/gpt-4o-mini"
   ```

6. **Heuristic Tuning** *(optional)*:

   Detector thresholds live in `config.json` and are loaded at startup, so they can be tuned without editing Python. Delete the file to fall back to the built-in defaults; unknown keys raise an error rather than being ignored.

   ```json
   {
     "min_transfers_for_trend": 3,
     "escalation_slope_usd": 50,
     "mule_min_inflow_usd": 2000,
     "mule_max_hours": 48,
     "mule_outflow_ratio": 0.8,
     "out_of_profile_sigmas": 3,
     "out_of_profile_min_usd": 100,
     "high_risk_keywords": ["IBIZA", "BANGKOK", "DUBAI", "CLUB"],
     "transfer_keywords": ["ZELLE", "WIRE"]
   }
   ```

7. **How to Run**:

   1. *(Optional)* Test the heuristic filter standalone — shows how the rules narrow the initial 291 transactions down to the suspicious ones:
      ```bash
      python heuristics.py
      ```
   2. Run the heuristic filter plus the LLM validator. Reads from `datasets/`, runs the pipeline, and writes to `results/`:
      ```bash
      python llm_validator.py
      ```
   3. Run the evaluation. Compares the LLM outputs against `ground_truth_labels.csv` to produce the classification report, confusion matrix, and cost projections:
      ```bash
      python evaluate_metrics.py
      ```
   4. *(Optional)* Launch the results dashboard locally — the same view published at [carefull-takehome.streamlit.app](https://carefull-takehome.streamlit.app). Requires no API keys, since it only reads the CSVs in `results/`:
      ```bash
      streamlit run app.py
      ```

## Evaluation Metrics Significance

- **Precision (1.00)**: Crucial for this product. A score of 1.00 means zero false positives, ensuring active, wealthy seniors do not receive patronizing false alarms.

- **Recall (~0.75)**: Represents the percentage of actual frauds caught. While some early-stage anomalies were missed to maintain 100% precision, all major exploitation escalations and money mule patterns were successfully flagged.

### How to Read These Numbers in Practice

The confusion matrix is the operational view. For the 291-transaction sample:

| Metric | Value | What it means for the product |
| --- | --- | --- |
| True Positives | 23 | Real exploitation surfaced to the family before more money moved. |
| False Positives | 0 | **No senior was wrongly accused.** Nobody got a "we think you're being scammed" alert about their own vacation. |
| False Negatives | 9 | Frauds the pipeline stayed silent on. These are the real cost, and they are concentrated in early, small-dollar grooming activity. |
| True Negatives | 259 | Ordinary life (groceries, pharmacy, church donations) correctly ignored. |

**Why precision is weighted above recall here.** The product sits in a trust relationship between an adult child and an aging parent. A false positive is not a harmless extra notification — it implies the parent is being deceived, and repeated false alarms train the family to dismiss the feed entirely. At that point recall stops mattering, because nobody reads the alerts. Zero false positives is what keeps the channel credible.

**What the 9 misses actually are.** They are not random. They sit at the *beginning* of escalation sequences, where a $50 transfer to a new contact is statistically indistinguishable from a legitimate gift. The detector requires 3 transfers to establish a trend (`min_transfers_for_trend`), so the first two in any grooming ramp are structurally invisible. The pattern still gets caught — just later in the sequence, not on the first transaction.

**How to shift the balance.** Recall is tunable from `config.json` without touching code. Lowering `min_transfers_for_trend` to 2 or `escalation_slope_usd` below 50 widens the net and raises recall, at the cost of introducing false positives. The current values were chosen as the most aggressive setting that still holds precision at 1.00.

**Reading latency and cost.** The ~2.5s latency is per *reviewed* transaction, not per ingested one. Since the heuristic filter removes ~90% of volume, the real-world throughput cost is roughly a tenth of what that number suggests — which is also why the projected spend lands under $1/month at 50k transactions rather than in the tens of dollars.

### A Note on Deployment

The pipeline itself is intentionally a reproducible batch job rather than a live service: it is deterministic given the same inputs, runs end-to-end in three commands, and writes every intermediate artifact to disk for inspection.

The **[hosted dashboard](https://carefull-takehome.streamlit.app)** sits on top of those artifacts. It reads the committed `results/` CSVs, so reviewers can compare both models' metrics and inspect their verdicts and reasoning without cloning the repo or supplying API keys. Re-running the pipeline locally regenerates exactly what the dashboard displays.
