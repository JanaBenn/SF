from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional
from domain.segment import Segment

class ReturnLookupService:
    """
    Baut und nutzt die Segment-Return-Lookup-Tabelle:
    Spalten: Gender, Income_Label, Discount_Used, Age_Group, Purchase_Category,
             total_purchases, total_returns, Return_%
    """

    def __init__(self, df_prepared: pd.DataFrame):
        # erwartet: df schon mit Age_Group und Income_Label
        self.df = df_prepared.copy()

    # ---------- Build ----------
    def build_lookup(self) -> pd.DataFrame:
        grouped = self.df.groupby(
            ["Gender", "Income_Label", "Discount_Used", "Age_Group", "Purchase_Category"],
            observed=False
        ).agg(
            total_purchases=("Return_Rate", "count"),
            total_returns=("Return_Rate", "sum")
        ).reset_index()

        grouped["Return_%"] = (grouped["total_returns"] / grouped["total_purchases"]) * 100

        # Serialisierungsfreundlich
        for col in ["Age_Group", "Income_Label", "Gender", "Purchase_Category"]:
            grouped[col] = grouped[col].astype(str)

        return grouped

    @staticmethod
    def save_csv(lookup_df: pd.DataFrame, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lookup_df.to_csv(path, index=False)

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    # ---------- Query ----------
    @staticmethod
    def filter_lookup(
        lookup_df: pd.DataFrame,
        gender: Optional[str] = None,
        income_label: Optional[str] = None,
        discount_used: Optional[bool] = None,
        age_groups: Optional[Iterable[str]] = None,
        categories: Optional[Iterable[str]] = None
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

    @staticmethod
    def topk_categories_for_segment(
        lookup_df: pd.DataFrame,
        segment: Segment,
        age_group: str,
        k: int = 5,
        highest_first: bool = True,
    ) -> pd.DataFrame:
        seg = ReturnLookupService.filter_lookup(
            lookup_df,
            gender=segment.gender,
            income_label=segment.income_label,
            discount_used=segment.discount_used,
            age_groups=[age_group]
        )
        if seg.empty:
            return seg
        return (
            seg.sort_values("Return_%", ascending=not highest_first)
               .head(k)
               .reset_index(drop=True)
        )

    @staticmethod
    def global_topk(
        lookup_df: pd.DataFrame,
        gender_whitelist=("Male", "Female"),
        k: int = 5
    ) -> pd.DataFrame:
        """
        Deine gewünschte globale Top‑5:
        - nur Male/Female
        - Primär sortiert nach Return_% (desc)
        - Sekundär nach total_purchases (desc)
        """
        df = lookup_df.copy()
        df = df[df["Gender"].isin(gender_whitelist)]
        out = (
            df.sort_values(["Return_%", "total_purchases"], ascending=[False, False])
              .head(k)
              .reset_index(drop=True)
        )
        # Bonus: ein Segment-Label hinzupacken
        out["Segment"] = (
            out["Gender"] + " | " + out["Income_Label"] +
            " | Disc=" + out["Discount_Used"].astype(str)
        )
        return out

    @staticmethod
    def pivot_segment_age_product(
        lookup_df: pd.DataFrame,
        segment: Segment
    ) -> pd.DataFrame:
        seg = ReturnLookupService.filter_lookup(
            lookup_df,
            gender=segment.gender,
            income_label=segment.income_label,
            discount_used=segment.discount_used
        )
        if seg.empty:
            return pd.DataFrame()

        pivot = seg.pivot_table(
            index="Age_Group",
            columns="Purchase_Category",
            values="Return_%",
            aggfunc="mean",
            observed=False
        )
        return pivot

    @staticmethod
    def summarize_by_age(
        lookup_df: pd.DataFrame,
        segments: Optional[list[Segment]] = None
    ) -> pd.DataFrame:
        """
        Kompakte Kennzahlen je Segment & Age_Group (ohne Mindestanzahl-Filter).
        """
        if segments is None:
            segments = [
                Segment("Female", "High", True),   Segment("Female", "High", False),
                Segment("Female", "Middle", True), Segment("Female", "Middle", False),
                Segment("Male", "High", True),     Segment("Male", "High", False),
                Segment("Male", "Middle", True),   Segment("Male", "Middle", False),
            ]

        rows = []
        for seg in segments:
            df = ReturnLookupService.filter_lookup(
                lookup_df,
                gender=seg.gender,
                income_label=seg.income_label,
                discount_used=seg.discount_used
            )
            if df.empty:
                continue
            agg = (
                df.groupby(["Gender", "Income_Label", "Discount_Used", "Age_Group"], observed=False)
                  .agg(
                      base_purchases=("total_purchases", "sum"),
                      n_categories=("Return_%", "count"),
                      mean_Return_pct=("Return_%", "mean"),
                      max_Return_pct=("Return_%", "max"),
                      min_Return_pct=("Return_%", "min"),
                  )
                  .reset_index()
            )
            rows.append(agg)

        if not rows:
            return pd.DataFrame(
                columns=["Gender","Income_Label","Discount_Used","Age_Group",
                         "base_purchases","n_categories","mean_Return_pct","max_Return_pct","min_Return_pct"]
            )

        out = pd.concat(rows, ignore_index=True)
        return out.sort_values(["Gender","Income_Label","Discount_Used","Age_Group"]).reset_index(drop=True)
