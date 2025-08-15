from __future__ import annotations
import pandas as pd

class RuleEngine:
    """
    Enthält Prüf- und Filterregeln für Lookups (Return_% & Basis) und Modell-ORs.
    """

    def __init__(self, min_base: int = 3, cap_or: float = 10.0):
        self.min_base = int(min_base)
        self.cap_or = float(cap_or)

    # ---- Lookup-Tabellen ----
    def mark_low_base(self, lookup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Spalte 'low_base' hinzu: True, wenn total_purchases < min_base.
        """
        out = lookup_df.copy()
        if "total_purchases" not in out.columns:
            raise KeyError("'total_purchases' fehlt im DataFrame.")
        out["low_base"] = out["total_purchases"] < self.min_base
        return out

    def filter_low_base(self, lookup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Entfernt alle Zeilen mit low_base == True (falls Spalte vorhanden).
        """
        if "low_base" in lookup_df.columns:
            return lookup_df[~lookup_df["low_base"]].reset_index(drop=True)
        return lookup_df.reset_index(drop=True)

    def filter_by_return_pct(self, lookup_df: pd.DataFrame, min_return_pct: float) -> pd.DataFrame:
        """
        Entfernt alle Zeilen mit Return_% < min_return_pct.
        """
        if "Return_%" not in lookup_df.columns:
            raise KeyError("'Return_%' fehlt im DataFrame.")
        return lookup_df[lookup_df["Return_%"] >= min_return_pct].reset_index(drop=True)

    def flag_high_risk(self, lookup_df: pd.DataFrame, min_return_pct=80.0) -> pd.DataFrame:
        """
        Liefert nur Zeilen mit hohem Rückgaberisiko und ausreichender Basis.
        """
        df = self.mark_low_base(lookup_df)
        return df[(df["Return_%"] >= min_return_pct) & (~df["low_base"])].reset_index(drop=True)

    # ---- Modell-ORs ----
    def clip_odds(self, or_df: pd.DataFrame) -> pd.DataFrame:
        """
        Klemmt OR-Werte in den Bereich [1/cap_or, cap_or].
        """
        out = or_df.copy()
        if "OR" not in out.columns:
            raise KeyError("'OR' fehlt im DataFrame.")
        out["OR"] = out["OR"].clip(lower=1.0 / self.cap_or, upper=self.cap_or)
        return out