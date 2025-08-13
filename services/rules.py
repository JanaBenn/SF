from __future__ import annotations
import pandas as pd

class RuleEngine:
    """
    Beispiel-Regeln, um aus Lookup „Risiken“ zu markieren.
    """

    def flag_high_risk(self, lookup_df: pd.DataFrame, min_return_pct=80.0, min_total_purchases=3) -> pd.DataFrame:
        df = lookup_df.copy()
        return df[(df["Return_%"] >= min_return_pct) & (df["total_purchases"] >= min_total_purchases)]
