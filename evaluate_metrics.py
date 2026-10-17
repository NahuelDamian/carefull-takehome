import json
import os

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from heuristics import _read_csv_streamed

# Risk levels the LLM can emit that count as an actionable alert.
ALERT_RISK_LEVELS = {'high', 'medium'}
# Label in ground_truth_labels.csv that means "not fraud".
BENIGN_LABEL = 'benign'


def load_and_prepare_data(results_file, truth_file, total_transactions_file):
    """
    Builds the evaluation frame: every transaction in the dataset, with its
    ground-truth label and the LLM verdict (absent when the heuristic filtered it out).

    Args:
        results_file: CSV produced by llm_validator.py for a single provider.
        truth_file: CSV with the `ground_truth_label` for each transaction.
        total_transactions_file: The full transaction universe being scored.

    Returns:
        DataFrame with `true_label` and `predicted_label` as 0/1 columns.

    Raises:
        FileNotFoundError: If any input path does not exist.
        KeyError: If a required column is missing from any input.
        ValueError: If an input file is empty.
    """
    for file_path in [results_file, truth_file, total_transactions_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    # Streamed in bounded chunks so peak memory stays flat as these files grow.
    results = _read_csv_streamed(results_file)
    truth = _read_csv_streamed(truth_file)
    all_tx = _read_csv_streamed(total_transactions_file)
    
    # Validate structure up front so a schema change fails loudly here rather than
    # as an opaque KeyError/IndexError deep inside the metric calculations.
    datasets = [
        (all_tx, 'transactions', total_transactions_file, ['transaction_id']),
        (truth, 'truth labels', truth_file, ['transaction_id', 'ground_truth_label']),
        (results, 'LLM results', results_file, ['transaction_id', 'risk_level', 'latency_sec']),
    ]
    for df, name, path, required_columns in datasets:
        if df.empty:
            raise ValueError(f"The {name} file is empty: {path}")
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise KeyError(f"Missing column(s) {missing} in {name} file: {path}")

    # Duplicated IDs would fan out the merge and silently inflate the metrics.
    duplicate_count = results['transaction_id'].duplicated().sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicated transaction_id rows in {results_file}")

    # Start from the full universe of transactions so that anything the heuristic
    # filtered out still counts as a prediction of "benign" (and as a miss if it was fraud).
    df = pd.merge(all_tx[['transaction_id']], truth[['transaction_id', 'ground_truth_label']], on='transaction_id', how='left')
    df = pd.merge(df, results[['transaction_id', 'risk_level', 'latency_sec']], on='transaction_id', how='left')

    # Map ground truth to binary: anything not explicitly benign is fraud (1).
    df['true_label'] = (df['ground_truth_label'].fillna(BENIGN_LABEL).str.lower() != BENIGN_LABEL).astype(int)

    # Map Predictions to binary
    # If the LLM said 'high' or 'medium', it's an alert (1)
    # If the LLM said 'low', or if it was filtered by the heuristic (NaN), it's benign (0)
    def map_prediction(risk):
        if pd.isna(risk):
            return 0
        return 1 if str(risk).lower() in ALERT_RISK_LEVELS else 0
        
    df['predicted_label'] = df['risk_level'].apply(map_prediction)
    return df

def calculate_costs(provider, total_latency, total_tx_processed, monthly_volume=50000):
    """
    Projects the monthly LLM spend at production volume.

    Args:
        provider: 'openrouter' or 'groq'; selects the per-transaction price.
        total_latency: Accumulated seconds spent on LLM calls (reporting only).
        total_tx_processed: Count of transactions flagged as alerts (reporting only).
        monthly_volume: Expected total transactions per month before filtering.

    Returns:
        Dict with the per-transaction price and the projected monthly cost.
    """
    # Approximate cost estimates per transaction based on current LLMs
    # GPT-4o-mini costs ~$0.00015 per average prompt/completion for this case.
    # Llama-3-70B via Groq is extremely low cost in free tier, but we assume commercial cost ~$0.0006
    cost_per_tx = 0.00015 if provider == 'openrouter' else 0.0006
    
    # We only process ~10% of transactions thanks to the heuristic filter
    filter_rate = 0.10 
    tx_to_llm_monthly = monthly_volume * filter_rate
    
    monthly_cost = tx_to_llm_monthly * cost_per_tx
    
    return {
        "cost_per_llm_tx": f"${cost_per_tx:.5f}",
        "projected_monthly_cost": f"${monthly_cost:.2f} (processing {int(tx_to_llm_monthly)} txs with LLM)",
        # Raw numbers alongside the display strings, so callers that need to chart
        # or compare costs do not have to parse the formatted text back out.
        "cost_per_llm_tx_usd": cost_per_tx,
        "projected_monthly_cost_usd": monthly_cost,
    }


def compute_metrics(provider_name, results_path, truth_path, tx_path):
    """
    Scores one provider and returns every metric as data instead of printing it.

    This is the single source of truth for the evaluation: both the CLI report
    and the Streamlit dashboard consume it, so the two can never drift apart.

    Args:
        provider_name: Provider key used for pricing and display.
        results_path: CSV with that provider's LLM verdicts.
        truth_path: CSV with the ground-truth labels.
        tx_path: CSV with the full transaction universe.

    Returns:
        Dict with the scored DataFrame, classification report (dict and text),
        confusion matrix, latency summary, and cost projection. `report`,
        `report_text` and `confusion_matrix` are None if sklearn could not
        compute them.

    Raises:
        FileNotFoundError, KeyError, ValueError: Propagated from load_and_prepare_data.
    """
    df = load_and_prepare_data(results_path, truth_path, tx_path)

    report, report_text, matrix = None, None, None
    try:
        # target_names: 0 = Normal, 1 = Fraud
        report_args = {
            'labels': [0, 1],
            'target_names': ['Normal', 'Fraud'],
            'zero_division': 0,
        }
        report = classification_report(
            df['true_label'], df['predicted_label'], output_dict=True, **report_args
        )
        report_text = classification_report(
            df['true_label'], df['predicted_label'], **report_args
        )
    except ValueError as e:
        report_error = str(e)
    else:
        report_error = None

    try:
        # labels=[0, 1] forces a 2x2 matrix even if one class is absent from the data,
        # which would otherwise collapse the shape and raise an IndexError downstream.
        matrix = confusion_matrix(df['true_label'], df['predicted_label'], labels=[0, 1])
    except ValueError as e:
        matrix_error = str(e)
    else:
        matrix_error = None

    # Only transactions that actually reached the LLM have a latency; the ones the
    # heuristic filtered out are NaN and must not drag the average toward zero.
    reviewed = df['latency_sec'].dropna()

    return {
        'provider': provider_name,
        'data': df,
        'report': report,
        'report_text': report_text,
        'report_error': report_error,
        'confusion_matrix': matrix,
        'matrix_error': matrix_error,
        'avg_latency_sec': float(reviewed.mean()) if not reviewed.empty else None,
        'reviewed_count': int(reviewed.count()),
        'total_transactions': int(len(df)),
        'alert_count': int((df['predicted_label'] == 1).sum()),
        'costs': calculate_costs(provider_name, reviewed.sum(), int((df['predicted_label'] == 1).sum())),
    }

def evaluate_model(provider_name, results_path, truth_path, tx_path):
    """
    Prints the full evaluation for one provider: classification report,
    confusion matrix, average latency, and projected cost.

    Args:
        provider_name: Provider key used for pricing and display.
        results_path: CSV with that provider's LLM verdicts.
        truth_path: CSV with the ground-truth labels.
        tx_path: CSV with the full transaction universe.

    Returns:
        The scored DataFrame, for further inspection by the caller.
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING MODEL: {provider_name.upper()}")
    print(f"{'='*40}")
    
    metrics = compute_metrics(provider_name, results_path, truth_path, tx_path)
    
    # Metrics
    print("\n1. Classification Report (Precision, Recall, F1):")
    if metrics['report_text'] is None:
        print(f"Could not compute the classification report: {metrics['report_error']}")
    else:
        print(metrics['report_text'])
    
    print("\n2. Confusion Matrix:")
    cm = metrics['confusion_matrix']
    if cm is None:
        print(f"Could not compute the confusion matrix: {metrics['matrix_error']}")
    else:
        print(f"True Negatives (Normal -> Normal): {cm[0][0]}")
        print(f"False Positives (Normal -> Fraud): {cm[0][1]} <- False alarms")
        print(f"False Negatives (Fraud -> Normal): {cm[1][0]} <- Fraud we missed")
        print(f"True Positives (Fraud -> Fraud): {cm[1][1]} <- Fraud caught")
    
    # Latency and Costs
    print(f"\n3. Performance and Cost (50k tx/month projection):")
    if metrics['avg_latency_sec'] is None:
        print("No transactions reached the LLM, so there is no latency to report.")
    else:
        print(f"Average latency per transaction sent to LLM: {metrics['avg_latency_sec']:.2f} seconds")
    costs = metrics['costs']
    print(json.dumps({
        'cost_per_llm_tx': costs['cost_per_llm_tx'],
        'projected_monthly_cost': costs['projected_monthly_cost'],
    }, indent=2))
    
    return metrics['data']

if __name__ == "__main__":
    truth_file = "datasets/ground_truth_labels.csv"
    tx_file = "datasets/transactions.csv"
    
    # Evaluate OpenRouter (GPT-4o-mini)
    df_or = evaluate_model("openrouter", "results/results_openrouter.csv", truth_file, tx_file)
    
    # Evaluate Groq (gpt-oss-20b)
    df_groq = evaluate_model("groq", "results/results_groq.csv", truth_file, tx_file)
