from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, List, Optional
from domain.segment import Segment


# ---------- Normalisierungen (wie in deinen Services) ----------

def _categorize_age(df: pd.DataFrame) -> pd.DataFrame:
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
    out = df.copy()
    if pd.api.types.is_numeric_dtype(out["Income_Level"]):
        out["Income_Label"] = out["Income_Level"].map({1: "Low", 2: "Middle", 3: "High"})
    else:
        canon = (
            out["Income_Level"].astype(str).str.strip()
            .str.replace(r"\s+", " ", regex=True).str.title()
        )
        canon = canon.where(canon.isin(["Low", "Middle", "High"]), other=np.nan)
        out["Income_Label"] = canon
    return out

def _normalize_boolish(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    mapping = {
        "1": True, "0": False,
        "true": True, "false": False,
        "t": True, "f": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(False).astype(bool)

def _ads_status(series: pd.Series) -> pd.Series:
    """High/Medium/Low -> 'vorhanden', None/leer -> 'keine'."""
    v = series.astype(str).str.strip().str.lower()
    present = v.isin({"high", "medium", "low"})
    return pd.Series(np.where(present, "vorhanden", "keine"), index=series.index)

# ---------- Recommender ----------

class CampaignRecommender:
    """
    Baut Empfehlungen aus Lookup + Rohdaten.
    - arbeitet auf ALLEN Segmenten (Female/Male × High/Middle × Discount T/F),
      außer du gibst explizit eine Liste von Segmenten rein.
    - Empfehlungstext ist modular:
        * Rückversandoptionen anpassen (immer)
        * Discount vorschlagen (wenn Discount_Used == False)
        * Membership-Programm vorschlagen (wenn Loyalty == 'keine')
        * Ads prüfen (wenn Ads == 'vorhanden')
    """

    def __init__(self, use_high_risk: bool = True):
        self.use_high_risk = use_high_risk

    def _build_for_one_segment(
        self,
        lookup_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        seg: Segment,
    ) -> pd.DataFrame:
        """Empfehlungstabelle für EIN Segment (alle Altersgruppen & Kategorien)."""
        tmp = raw_df.copy()
        tmp = _categorize_age(tmp)
        tmp = _categorize_income(tmp)

        # Discount & Loyalty & Ads
        if "Discount_Used" in tmp.columns and tmp["Discount_Used"].dtype != bool:
            tmp["Discount_Used"] = _normalize_boolish(tmp["Discount_Used"])

        loyalty_bool = _normalize_boolish(tmp.get("Customer_Loyalty_Program_Member", pd.Series(index=tmp.index)))
        tmp["Loyalty_Status"] = np.where(loyalty_bool, "vorhanden", "keine")
        tmp["Ads_Status"] = _ads_status(tmp.get("Engagement_with_Ads", pd.Series(index=tmp.index, dtype=object)))

        # Segment-Fokus
        seg_raw = tmp[
            (tmp["Gender"] == seg.gender)
            & (tmp["Income_Label"] == seg.income_label)
            & (tmp["Discount_Used"] == seg.discount_used)
        ].dropna(subset=["Age_Group", "Purchase_Category"])

        # Kontext je (Age_Group, Purchase_Category)
        if seg_raw.empty:
            return pd.DataFrame(
                columns=[
                    "Gender","Income_Label","Discount_Used","Age_Group","Purchase_Category",
                    "total_purchases","total_returns","Return_%",
                    "Ads_Status","Loyalty_Status","Recommendation","Segment"
                ]
            )

        ctx = (
            seg_raw.groupby(["Gender","Income_Label","Discount_Used","Age_Group","Purchase_Category"], observed=True)
            .agg(
                Ads_Status=("Ads_Status", lambda s: "vorhanden" if (s == "vorhanden").any() else "keine"),
                Loyalty_Status=("Loyalty_Status", lambda s: "vorhanden" if (s == "vorhanden").any() else "keine"),
            )
            .reset_index()
        )

        seg_lookup = lookup_df[
            (lookup_df["Gender"] == seg.gender)
            & (lookup_df["Income_Label"] == seg.income_label)
            & (lookup_df["Discount_Used"] == seg.discount_used)
        ].copy()

        if self.use_high_risk and "high_risk" in seg_lookup.columns:
            seg_lookup = seg_lookup[seg_lookup["high_risk"]]

        merged = pd.merge(
            seg_lookup, ctx,
            on=["Gender","Income_Label","Discount_Used","Age_Group","Purchase_Category"],
            how="left", validate="one_to_one"
        )

        merged["Segment"] = f"{seg.gender} | {seg.income_label} | Disc={seg.discount_used}"

        def _build_text(row: pd.Series) -> str:
            parts = ["Rückversandoptionen anpassen"]
            if not bool(row.get("Discount_Used", False)):
                parts.append("Discount vorschlagen")
            if str(row.get("Loyalty_Status","keine")).strip().lower() == "keine":
                parts.append("Membership-Programm vorschlagen")
            if str(row.get("Ads_Status","keine")).strip().lower() == "vorhanden":
                parts.append("Ads prüfen")
            return " | ".join(parts)

        merged["Recommendation"] = merged.apply(_build_text, axis=1)

        cols = [
            "Gender","Income_Label","Discount_Used","Age_Group","Purchase_Category",
            "total_purchases","total_returns","Return_%",
            "Ads_Status","Loyalty_Status",
            "Recommendation","Segment"
        ]
        return merged[cols].reset_index(drop=True)

    def build_recommendations_all(
        self,
        lookup_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        segments: Optional[Iterable[Segment]] = None,
        return_pct_min: float = 51.0,
    ) -> pd.DataFrame:
        """
        Baut eine kombinierte Empfehlungstabelle für ALLE Segmente
        und filtert anschließend auf Return_% > return_pct_min.
        Sortierung übernimmt später der Plot/Export.
        """
        if segments is None:
            # wie im Modeling: nur High/Middle, beide Geschlechter, Disc T/F
            segments = [
                Segment("Female","High",True),   Segment("Female","High",False),
                Segment("Female","Middle",True), Segment("Female","Middle",False),
                Segment("Male","High",True),     Segment("Male","High",False),
                Segment("Male","Middle",True),   Segment("Male","Middle",False),
            ]

        frames: List[pd.DataFrame] = []
        for s in segments:
            one = self._build_for_one_segment(lookup_df, raw_df, s)
            if not one.empty:
                frames.append(one)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "Gender","Income_Label","Discount_Used","Age_Group","Purchase_Category",
                    "total_purchases","total_returns","Return_%",
                    "Ads_Status","Loyalty_Status","Recommendation","Segment"
                ]
            )

        out = pd.concat(frames, ignore_index=True)

        # Filter 1) Return_% > 51
        out = out[out["Return_%"] > float(return_pct_min)].copy()

        # Robust: Age_Group als sortierbare Kategorie setzen
        age_order = ["<=20","21–30","31–40","41–50","51–60"]
        out["Age_Group"] = pd.Categorical(out["Age_Group"], categories=age_order, ordered=True)

        return out.reset_index(drop=True)