import json
import os
import re
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

DEFAULT_CHUNK_SIZE = 50_000        # Rows per chunk when streaming large CSVs
DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class HeuristicConfig:
    """
    Tunable thresholds for the three detectors.

    Defaults are the values validated against the sample dataset. They can be
    overridden per-run without touching the code, either by passing a
    HeuristicConfig instance or by editing `config.json` (see `from_json`).
    """
    min_transfers_for_trend: int = 3        # Transfers to a recipient before a trend is meaningful
    escalation_slope_usd: float = 50        # Min. $ increase per transfer for a relevant escalation
    mule_min_inflow_usd: float = 2000       # Inflow size that makes a fast in/out cycle suspicious
    mule_max_hours: float = 48              # Max. window between inflow and outflow
    mule_outflow_ratio: float = 0.80        # Outflow must move at least this share of the inflow
    out_of_profile_sigmas: float = 3        # Std deviations above the user's mean spend
    out_of_profile_min_usd: float = 100     # Floor to avoid flagging tiny amounts on low-variance users
    high_risk_keywords: list = field(default_factory=lambda: ['IBIZA', 'BANGKOK', 'DUBAI', 'CLUB'])
    transfer_keywords: list = field(default_factory=lambda: ['ZELLE', 'WIRE'])

    @classmethod
    def from_json(cls, path=DEFAULT_CONFIG_PATH):
        """
        Loads thresholds from a JSON file, falling back to defaults when the file
        is absent. Unknown keys are rejected so typos surface immediately instead
        of silently leaving a threshold at its default.
        """
        if not os.path.exists(path):
            return cls()

        with open(path, 'r', encoding='utf-8') as config_file:
            overrides = json.load(config_file)

        valid_keys = {f.name for f in fields(cls)}
        unknown = set(overrides) - valid_keys
        if unknown:
            raise KeyError(f"Unknown config keys in {path}: {', '.join(sorted(unknown))}")

        return cls(**overrides)


def _read_csv_streamed(path, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Reads a CSV in bounded chunks instead of a single full-file load.

    Pandas allocates a large intermediate buffer when parsing a whole file at once;
    streaming keeps peak memory proportional to `chunk_size` rather than file size.
    Passing chunk_size=None falls back to a plain single-shot read.
    """
    if chunk_size is None:
        return pd.read_csv(path)

    chunks = [chunk for chunk in pd.read_csv(path, chunksize=chunk_size)]
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


class TransactionAnalyzer:
    def __init__(self, transactions_path, profiles_path, chunk_size=DEFAULT_CHUNK_SIZE, config=None):
        """
        Loads and prepares the data for analysis.

        Args:
            transactions_path: CSV with one row per transaction.
            profiles_path: CSV with one row per user (biographical context).
            chunk_size: Rows read per chunk. Use None to disable streaming.
            config: HeuristicConfig with the detector thresholds. Defaults to
                `config.json` if present, otherwise to the built-in values.
        """
        if not os.path.exists(transactions_path) or not os.path.exists(profiles_path):
            raise FileNotFoundError(f"Missing dataset files. Check paths: {transactions_path}, {profiles_path}")

        self.config = config or HeuristicConfig.from_json()

        # Transactions are the unbounded side of the dataset, so they are streamed.
        # Profiles are small (one row per user) and read in a single pass.
        self.tx = _read_csv_streamed(transactions_path, chunk_size)
        self.users = pd.read_csv(profiles_path)

        if 'user_id' not in self.tx.columns or 'user_id' not in self.users.columns:
            raise KeyError("Missing 'user_id' column required for merging.")

        self.tx['date'] = pd.to_datetime(self.tx['date'])
        self.df = pd.merge(self.tx, self.users, on='user_id', how='left')
        self.df = self.df.sort_values(by=['user_id', 'date'])

    def enrich_transactions(self):
        """
        Phase 2: Enrichment.
        Extracts the recipient name from transfer descriptions so that repeated
        payments to the same person can be grouped, even when the raw bank
        description varies (e.g. "ZELLE PAYMENT TO K RICHARDS" -> "K RICHARDS").
        """
        def extract_entity(desc):
            if pd.isna(desc) or not isinstance(desc, str):
                return None
            desc_upper = desc.upper()
            # Only P2P rails carry a counterparty name worth isolating.
            if any(keyword in desc_upper for keyword in self.config.transfer_keywords):
                # Strip rail/action keywords so only the counterparty name remains.
                clean_desc = re.sub(r'(ZELLE|WIRE|PAYMENT|TRANSFER|TO|FROM|PMT)\s+', '', desc_upper)
                return clean_desc.strip()
            return None

        self.df['recipient_entity'] = self.df['raw_description'].apply(extract_entity)
        return self.df

    def detect_escalating_transfers(self):
        """
        Pattern 1: Escalating transfers to an unknown.
        Searches for recurring recipients where the average amount increases over time.
        """
        flags = []
        # We only analyze debits that have an extracted recipient
        transfers = self.df[(self.df['type'] == 'debit') & (self.df['recipient_entity'].notnull())]
        
        for (user, recipient), group in transfers.groupby(['user_id', 'recipient_entity']):
            if len(group) >= self.config.min_transfers_for_trend:
                # Fit a first-degree polynomial (least-squares line) over the ordered
                # transfers: the slope is the average $ increase per transfer.
                # A positive, steep slope is the signature of a grooming/exploitation
                # ramp, where the scammer tests ever-larger amounts.
                group = group.sort_values('date')
                x = np.arange(len(group))
                y = group['amount'].values

                # len(set(y)) > 1 guards polyfit against a flat series (no trend to fit).
                if len(set(y)) > 1 and np.polyfit(x, y, 1)[0] > self.config.escalation_slope_usd:
                    flags.extend(group['transaction_id'].tolist())
                    
        return flags

    def detect_money_mule(self):
        """
        Pattern 2: Fast inflow and outflow (Money Mule).
        Searches for large deposits followed by large withdrawals in < 48 hours.
        """
        flags = []
        for user_id, group in self.df.groupby('user_id'):
            # Chronological order lets us compare each transaction with the next one
            # for the same user, which is where the pass-through pattern shows up.
            group = group.sort_values('date').reset_index(drop=True)
            for i in range(len(group) - 1):
                tx_in = group.iloc[i]
                tx_out = group.iloc[i+1]
                
                # We verify if it is a credit followed by a debit
                if tx_in['type'] == 'credit' and tx_out['type'] == 'debit':
                    # If the credit is large enough to be worth laundering
                    if tx_in['amount'] >= self.config.mule_min_inflow_usd:
                        # If it occurred inside the fast pass-through window
                        time_diff = (tx_out['date'] - tx_in['date']).total_seconds() / 3600
                        if 0 <= time_diff <= self.config.mule_max_hours:
                            # Money mules forward most of what they receive, keeping only a cut
                            if tx_out['amount'] >= (tx_in['amount'] * self.config.mule_outflow_ratio):
                                flags.extend([tx_in['transaction_id'], tx_out['transaction_id']])
        return list(set(flags))

    def detect_out_of_profile(self):
        """
        Pattern 3: Out-of-profile spending.
        Detects unusually high transactions compared to the user's average.
        """
        flags = []
        for user_id, group in self.df.groupby('user_id'):
            debits = group[group['type'] == 'debit']
            if len(debits) == 0:
                continue
                
            # Baseline is per-user: a $2k charge is normal for one senior and alarming
            # for another, so the threshold is derived from their own spending history.
            mean_spend = debits['amount'].mean()
            std_spend = debits['amount'].std()

            # Heuristic threshold: N standard deviations above the user's own mean.
            threshold = mean_spend + (self.config.out_of_profile_sigmas * std_spend)
            
            for _, tx in debits.iterrows():
                # Flag 1: Mathematically atypical spending
                if tx['amount'] > threshold and tx['amount'] > self.config.out_of_profile_min_usd:
                    flags.append(tx['transaction_id'])
                # Flag 2: Keywords for international/late-night spending (simple heuristic for this dataset)
                elif any(word in str(tx['raw_description']).upper() for word in self.config.high_risk_keywords):
                    flags.append(tx['transaction_id'])
                    
        return list(set(flags))

    def run_pipeline(self):
        """Runs all detectors and marks suspicious transactions."""
        self.enrich_transactions()
        
        esc_flags = self.detect_escalating_transfers()
        mule_flags = self.detect_money_mule()
        oop_flags = self.detect_out_of_profile()
        
        # We unify all transactions marked by our heuristics
        all_flagged_ids = set(esc_flags + mule_flags + oop_flags)
        
        # We create a column indicating if the transaction should be sent to the LLM
        self.df['requires_llm_review'] = self.df['transaction_id'].isin(all_flagged_ids)
        
        # We return only those that the LLM should review
        return self.df[self.df['requires_llm_review']]

# --- Execution ---
if __name__ == "__main__":
    analyzer = TransactionAnalyzer("datasets/transactions.csv", "datasets/user_profiles.csv")
    suspicious_tx = analyzer.run_pipeline()
    
    print(f"Total transactions sent for LLM review: {len(suspicious_tx)}")
    print(suspicious_tx[['user_id', 'date', 'amount', 'type', 'raw_description']])