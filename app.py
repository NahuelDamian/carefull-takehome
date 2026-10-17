"""
Streamlit dashboard for the Carefull transaction alert system.

Reads the committed `results/` artifacts produced by llm_validator.py and renders
both the per-transaction LLM verdicts and the evaluation metrics, for every
provider that has results on disk. All scoring is delegated to evaluate_metrics
so the dashboard and the CLI report can never disagree.
"""
import os

import pandas as pd
import streamlit as st

from evaluate_metrics import compute_metrics

TRUTH_FILE = "datasets/ground_truth_labels.csv"
TX_FILE = "datasets/transactions.csv"

# Display name -> provider key used for pricing, plus where its results live.
# Adding a third provider only requires one more entry here.
PROVIDERS = {
    "Groq — openai/gpt-oss-20b": {
        "key": "groq",
        "results": "results/results_groq.csv",
    },
    "OpenRouter — GPT-4o-mini": {
        "key": "openrouter",
        "results": "results/results_openrouter.csv",
    },
}

VERDICT_COLUMNS = ['transaction_id', 'user_id', 'risk_level', 'confidence', 'reasoning']
RISK_LEVELS = ['high', 'medium', 'low']


@st.cache_data(show_spinner=False)
def load_provider(provider_key, results_path):
    """
    Scores one provider and loads its raw verdicts.

    Cached because the evaluation re-reads and merges three CSVs, which would
    otherwise run again on every widget interaction.

    Returns:
        Tuple of (metrics dict from compute_metrics, raw verdict DataFrame).
    """
    metrics = compute_metrics(provider_key, results_path, TRUTH_FILE, TX_FILE)
    return metrics, pd.read_csv(results_path)


def summarize(label, metrics):
    """Flattens one provider's metrics into a single row for the comparison table."""
    report = metrics['report']
    matrix = metrics['confusion_matrix']
    fraud = report['Fraud'] if report else {}
    return {
        "Model": label,
        "Precision": fraud.get('precision'),
        "Recall": fraud.get('recall'),
        "F1": fraud.get('f1-score'),
        "False positives": int(matrix[0][1]) if matrix is not None else None,
        "False negatives": int(matrix[1][0]) if matrix is not None else None,
        "Frauds caught": int(matrix[1][1]) if matrix is not None else None,
        "Avg latency (s)": metrics['avg_latency_sec'],
        "Cost / month": metrics['costs']['projected_monthly_cost_usd'],
    }


st.set_page_config(page_title="Carefull Fraud Detection", layout="wide")
st.title("🛡️ Carefull — Transaction Alert System")
st.caption(
    "A two-stage pipeline: deterministic heuristics narrow the transaction stream down to the "
    "suspicious minority, then an LLM reviews each candidate against the account holder's profile."
)

# Only surface providers whose results were actually committed. This keeps the
# dashboard usable when the pipeline has been run for one provider but not the other.
available = {
    label: config
    for label, config in PROVIDERS.items()
    if os.path.exists(config['results'])
}

if not available:
    st.error(
        "No results found in `results/`. Run `python llm_validator.py` to generate them, "
        "and make sure the output CSVs are committed if you are deploying this dashboard."
    )
    st.stop()

loaded, failed = {}, {}
for label, config in available.items():
    try:
        loaded[label] = load_provider(config['key'], config['results'])
    except (FileNotFoundError, KeyError, ValueError) as error:
        # One malformed provider file should not take the whole dashboard down.
        failed[label] = error

for label, error in failed.items():
    st.warning(f"Could not evaluate **{label}**: {error}")

if not loaded:
    st.stop()

# --- Model comparison -------------------------------------------------------
st.subheader("Model comparison")
st.caption(
    "Scored against `ground_truth_labels.csv` over the full transaction universe — transactions "
    "the heuristic filtered out count as predicted-benign, so these numbers are end-to-end."
)

comparison = pd.DataFrame([summarize(label, metrics) for label, (metrics, _) in loaded.items()])
st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Precision": st.column_config.NumberColumn(format="%.2f"),
        "Recall": st.column_config.NumberColumn(format="%.2f"),
        "F1": st.column_config.NumberColumn(format="%.2f"),
        "Avg latency (s)": st.column_config.NumberColumn(format="%.2f"),
        "Cost / month": st.column_config.NumberColumn(
            format="$%.2f", help="Projected spend at 50,000 transactions/month."
        ),
    },
)

with st.expander("How to read these numbers"):
    st.markdown(
        """
        **Precision is weighted above recall here.** The product sits in a trust relationship
        between an adult child and an aging parent. A false positive is not a harmless extra
        notification — it implies the parent is being deceived, and repeated false alarms train
        the family to dismiss the feed entirely. At that point recall stops mattering, because
        nobody reads the alerts.

        **The misses are not random.** They sit at the *beginning* of escalation sequences, where
        a small transfer to a new contact is statistically indistinguishable from a legitimate
        gift. The detector needs 3 transfers to establish a trend, so the first two in any
        grooming ramp are structurally invisible. The pattern still gets caught — later in the
        sequence, not on the first transaction.

        **Latency is per reviewed transaction, not per ingested one.** The heuristic filter
        removes ~90% of volume, which is why the projected spend lands under a few dollars a
        month rather than in the hundreds.
        """
    )

# --- Per-model detail -------------------------------------------------------
st.subheader("Detail by model")
selected = st.radio("Model", list(loaded.keys()), horizontal=True, label_visibility="collapsed")
metrics, verdicts = loaded[selected]
report = metrics['report']
matrix = metrics['confusion_matrix']

columns = st.columns(5)
columns[0].metric("Precision", f"{report['Fraud']['precision']:.2f}" if report else "n/a")
columns[1].metric("Recall", f"{report['Fraud']['recall']:.2f}" if report else "n/a")
columns[2].metric("F1", f"{report['Fraud']['f1-score']:.2f}" if report else "n/a")
columns[3].metric(
    "Avg latency",
    f"{metrics['avg_latency_sec']:.2f}s" if metrics['avg_latency_sec'] is not None else "n/a",
)
columns[4].metric(
    "Projected cost / month", f"${metrics['costs']['projected_monthly_cost_usd']:.2f}"
)

reviewed, total = metrics['reviewed_count'], metrics['total_transactions']
st.caption(
    f"The heuristic filter sent **{reviewed} of {total}** transactions to the LLM "
    f"({reviewed / total:.0%} of volume), of which **{metrics['alert_count']}** were escalated as alerts."
)

left, right = st.columns([1, 2])

with left:
    st.markdown("**Confusion matrix**")
    if matrix is None:
        st.info(f"Not available: {metrics['matrix_error']}")
    else:
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actually benign", "Actually fraud"],
                columns=["Predicted benign", "Predicted alert"],
            ),
            use_container_width=True,
        )
        st.caption(
            f"{matrix[0][1]} false positives (seniors wrongly flagged) · "
            f"{matrix[1][0]} false negatives (frauds missed)."
        )

with right:
    st.markdown("**LLM verdicts**")
    present_levels = [level for level in RISK_LEVELS if level in set(verdicts['risk_level'].str.lower())]
    chosen_levels = st.multiselect(
        "Risk level", present_levels, default=present_levels, key=f"levels_{selected}"
    )
    filtered = verdicts[verdicts['risk_level'].str.lower().isin(chosen_levels)]

    if filtered.empty:
        st.info("No transactions match the selected risk levels.")
    else:
        st.dataframe(
            filtered[VERDICT_COLUMNS],
            hide_index=True,
            use_container_width=True,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%d%%"
                ),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
        )
        st.caption("Click a reasoning cell to expand the model's full explanation.")
