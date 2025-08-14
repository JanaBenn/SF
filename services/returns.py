# services/returns.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Iterable, Tuple

# -------------------------------------------------------------------
# Hilfen
# -------------------------------------------------------------------

def _categorize_age(df: pd.DataFrame) -> pd.DataFrame:
    """Age -> Age_Group (<=20, 21–30, 31–40, 41–50, 51–60)."""
    bins = [0, 20, 30, 40, 50, 60]
    labels = ["<=20", "21–30", "31–40", "41–50", "51–60"]
    out = df.copy()
    out["Age_Group"] = pd.cut(
        out["Age"],
        bins=bins,
        labels=pd.Categorical(labels, categories=labels, ordered=True),
        right=False,
    )
    return out


def _categorize_income(df: pd.DataFrame) -> pd.DataFrame:
    """Income_Level -> Income_Label (robust für numerische Codes und bereits gelabelte Strings)."""
    out = df.copy()

    # Falls numerisch: 1/2/3 mappen
    if pd.api.types.is_numeric_dtype(out["Income_Level"]):
        income_map = {1: "Low", 2: "Middle", 3: "High"}
        out["Income_Label"] = out["Income_Level"].map(income_map)
    else:
        # Falls schon Strings: trimmen & vereinheitlichen
        canon = (
            out["Income_Level"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()  # z.B. "middle" -> "Middle"
        )
        # nur erlaubte Labels durchlassen
        canon = canon.where(canon.isin(["Low", "Middle", "High"]), other=np.nan)
        out["Income_Label"] = canon

    return out



def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten im DataFrame: {missing}")


def _auto_detect_return_col(df: pd.DataFrame) -> str:
    """Versucht die Return-Spalte automatisch zu finden."""
    candidates = [
        "Return_Rate",   # was wir bisher genutzt haben (0/1)
        "Return",        # alternative Bezeichnung
        "Returned",      # manchmal so benannt
        "is_return",     # booleans
        "ReturnFlag",    # weitere Variante
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Keine Return-Spalte gefunden. Erwartet eine der {candidates}. "
        f"Vorhandene Spalten: {list(df.columns)}"
    )


# -------------------------------------------------------------------
# Hauptservice
# -------------------------------------------------------------------

class ReturnLookupService:
    """
    Baut/verwaltet eine Lookup-Tabelle für Return-% pro
    (Gender, Income_Label, Discount_Used, Age_Group, Purchase_Category).
    """

    def __init__(self, df: pd.DataFrame, return_col: Optional[str] = None):
        """
        df: Rohdaten. Erwartet u.a. Age, Gender, Income_Level, Discount_Used, Purchase_Category, <Return-Spalte>
        return_col: Name der Return-Spalte (0/1). Wenn None -> Auto-Detection.
        """
        if return_col is None:
            return_col = _auto_detect_return_col(df)

        self.df = df.copy()
        self.return_col = return_col

        _ensure_cols(
            self.df,
            ["Age", "Gender", "Income_Level", "Discount_Used", "Purchase_Category", self.return_col],
        )

        # Kategorisieren
        self.df = _categorize_age(self.df)
        self.df = _categorize_income(self.df)

        # >>> WICHTIG: Nur Male/Female behalten
        self.df = self.df[self.df["Gender"].isin(["Male", "Female"])].copy()

        # Typen aufräumen
        self.df["Gender"] = self.df["Gender"].astype(str)
        self.df["Purchase_Category"] = self.df["Purchase_Category"].astype(str)

        # Discount_Used -> bool
        if self.df["Discount_Used"].dtype != bool:
            self.df["Discount_Used"] = (
                self.df["Discount_Used"]
                .astype(str).str.lower()
                .map({"1": True, "0": False, "true": True, "false": False})
                .fillna(False)
                .astype(bool)
            )

        # Return-Spalte binär (0/1) – zählt jede positive Zahl (inkl. 2) als Rückgabe
        self.df[self.return_col] = pd.to_numeric(self.df[self.return_col], errors="coerce").fillna(0.0)
        self.df[self.return_col] = (self.df[self.return_col] > 0).astype(float)

    def build_lookup(self) -> pd.DataFrame:
        """
        Aggregiert Return-% pro Segment * Age_Group * Kategorie.
        Spalten: Gender, Income_Label, Discount_Used, Age_Group, Purchase_Category,
                 total_purchases, total_returns, Return_%
        """
        # Nur vollständige Schlüsselzeilen
        df = self.df.dropna(subset=["Age_Group", "Income_Label", "Gender", "Purchase_Category"]).copy()

        print(f"[DEBUG] Starte build_lookup mit {len(df)} Zeilen")
        print("[DEBUG] Anzahl eindeutiger Purchase_Category:", df["Purchase_Category"].nunique())
        print("[DEBUG] Anzahl eindeutiger Age_Group:", df["Age_Group"].nunique())
        print("[DEBUG] Income_Label counts:", df["Income_Label"].value_counts(dropna=False).to_dict())
        print("[DEBUG] Gender counts:", df["Gender"].value_counts(dropna=False).to_dict())

        # >>> Nur beobachtete Kombinationen (observed=True)
        grouped = (
            df.groupby(
                ["Gender", "Income_Label", "Discount_Used", "Age_Group", "Purchase_Category"],
                observed=True,
            )[self.return_col]
            .agg(total_purchases="count", total_returns="sum")
            .reset_index()
        )

        # >>> 0er-Basen droppen (verhindert NaN in Return_%)
        grouped = grouped[grouped["total_purchases"] > 0].copy()

        grouped["Return_%"] = ((grouped["total_returns"] / grouped["total_purchases"]) * 100.0).round(1)

        # Strings (robuste Serialisierung)
        for c in ["Age_Group", "Income_Label", "Gender", "Purchase_Category"]:
            grouped[c] = grouped[c].astype(str)

        print("[DEBUG] grouped shape (nach Filter):", grouped.shape)
        print("[DEBUG] Beispielzeilen:\n", grouped.head(10))

        return grouped



    def save_csv(self, lookup_df: pd.DataFrame, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() != ".csv":
            p = p.with_suffix(".csv")
        lookup_df.to_csv(p, index=False)

    def save_parquet(self, lookup_df: pd.DataFrame, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() != ".parquet":
            p = p.with_suffix(".parquet")
        lookup_df.to_parquet(p, index=False)

    # ---------------- Queries ----------------

    @staticmethod
    def _add_segment_col(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Segment"] = (
            out["Gender"].astype(str)
            + " | "
            + out["Income_Label"].astype(str)
            + " | Disc="
            + out["Discount_Used"].astype(str)
        )
        return out

    @staticmethod
    def filter_lookup(
        lookup_df: pd.DataFrame,
        gender: Optional[str] = None,
        income_label: Optional[str] = None,
        discount_used: Optional[bool] = None,
        age_groups: Optional[Iterable[str]] = None,
        categories: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        df = lookup_df.copy()
        if gender is not None:
            df = df[df["Gender"] == gender]
        if income_label is not None:
            df = df[df["Income_Label"] == income_label]
        if discount_used is not None:
            df = df[df["Discount_Used"] == discount_used]
        if age_groups is not None:
            df = df[df["Age_Group"].isin(list(age_groups))]
        if categories is not None:
            df = df[df["Purchase_Category"].isin(list(categories))]
        return df.reset_index(drop=True)

    # Globale Top-K (über alle Segmente und Age_Groups)
    def global_topk(self, lookup_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        df = self._add_segment_col(lookup_df)
        # nach Return_% und Basis (total_purchases) sortieren
        out = df.sort_values(["Return_%", "total_purchases"], ascending=[False, False]).head(k)
        return out[
            [
                "Gender",
                "Income_Label",
                "Discount_Used",
                "Age_Group",
                "Purchase_Category",
                "total_purchases",
                "total_returns",
                "Return_%",
                "Segment",
            ]
        ].reset_index(drop=True)

    # Top-K in einem Segment + einer Altersgruppe
    def topk_categories_for_segment(
        self,
        lookup_df: pd.DataFrame,
        segment,  # domain.segment.Segment oder Tupel
        age_group: str,
        k: int = 5,
        ascending: bool = False,
    ) -> pd.DataFrame:
        if isinstance(segment, tuple) and len(segment) == 3:
            gender, income, disc = segment
        else:
            # erwarten API: Segment(gender, income_label, discount_used)
            gender, income, disc = segment.gender, segment.income_label, segment.discount_used

        seg = self.filter_lookup(
            lookup_df,
            gender=gender,
            income_label=income,
            discount_used=disc,
            age_groups=[age_group],
        )
        if seg.empty:
            return seg
        out = seg.sort_values(["Return_%", "total_purchases"], ascending=[ascending, True]).head(k)
        return out[
            [
                "Gender",
                "Income_Label",
                "Discount_Used",
                "Age_Group",
                "Purchase_Category",
                "total_purchases",
                "total_returns",
                "Return_%",
            ]
        ].reset_index(drop=True)

    # Pivot für Heatmap (Age x Category → Return_%)
    @staticmethod
    def pivot_segment_age_product(
        lookup_df: pd.DataFrame,
        segment,  # Segment oder Tupel
    ) -> pd.DataFrame:
        if isinstance(segment, tuple) and len(segment) == 3:
            gender, income, disc = segment
        else:
            gender, income, disc = segment.gender, segment.income_label, segment.discount_used

        seg = ReturnLookupService.filter_lookup(
            lookup_df, gender=gender, income_label=income, discount_used=disc
        )
        if seg.empty:
            return pd.DataFrame() 
        pivot = seg.pivot_table(
            index="Age_Group",
            columns="Purchase_Category",
            values="Return_%",
            aggfunc="mean",
            observed=False,
        )
        return pivot

    # Segment-Summary nach Age_Group
    @staticmethod
    def summarize_by_age(lookup_df: pd.DataFrame) -> pd.DataFrame:
        if lookup_df.empty:
            return pd.DataFrame(
                columns=[
                    "Gender",
                    "Income_Label",
                    "Discount_Used",
                    "Age_Group",
                    "base_purchases",
                    "n_categories",
                    "mean_Return_pct",
                    "max_Return_pct",
                    "min_Return_pct",
                ]
            )

        g = (
            lookup_df.groupby(
                ["Gender", "Income_Label", "Discount_Used", "Age_Group"], observed=False
            )
            .agg(
                base_purchases=("total_purchases", "sum"),
                n_categories=("Return_%", "count"),
                mean_Return_pct=("Return_%", "mean"),
                max_Return_pct=("Return_%", "max"),
                min_Return_pct=("Return_%", "min"),
            )
            .reset_index()
        )

        return g.sort_values(
            by=["Gender", "Income_Label", "Discount_Used", "Age_Group"]
        ).reset_index(drop=True)

    # ---- Heatmap-Helper (für visualisation.plot_segment_heatmap) ----

    @staticmethod
    def pivot_with_counts(
        lookup_df: pd.DataFrame,
        gender: str,
        income_label: str,
        discount_used: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        seg = ReturnLookupService.filter_lookup(
            lookup_df,
            gender=gender,
            income_label=income_label,
            discount_used=discount_used,
        )
        if seg.empty:
            return pd.DataFrame(), pd.DataFrame()

        pivot_val = seg.pivot_table(
            index="Age_Group",
            columns="Purchase_Category",
            values="Return_%",
            aggfunc="mean",
            observed=False,
        )
        pivot_n = seg.pivot_table(
            index="Age_Group",
            columns="Purchase_Category",
            values="total_purchases",
            aggfunc="sum",
            observed=False,
        )
        return pivot_val, pivot_n

    @staticmethod
    def build_heatmap_annotations(
        pivot_val: pd.DataFrame,
        pivot_n: pd.DataFrame,
        low_base_threshold: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Erzeugt Annotationen 'xx% (n=yy)' und Maske für kleine Basen."""
        annot = pd.DataFrame("", index=pivot_val.index, columns=pivot_val.columns)
        low_mask = pd.DataFrame(False, index=pivot_val.index, columns=pivot_val.columns)

        for r in pivot_val.index:
            for c in pivot_val.columns:
                val = pivot_val.loc[r, c]
                n = pivot_n.loc[r, c] if (r in pivot_n.index and c in pivot_n.columns) else np.nan
                if pd.isna(val):
                    text = ""
                else:
                    pct = f"{val:.0f}%"
                    text = pct if pd.isna(n) else f"{pct} (n={int(n)})"
                annot.loc[r, c] = text
                if not pd.isna(n) and int(n) < low_base_threshold:
                    low_mask.loc[r, c] = True

        return annot, low_mask
