from __future__ import annotations
import pandas as pd
from domain.segment import Segment

class CampaignRecommender:
    """
    Dummy-Recommender: liest Lookup + (später) Rules/ORs und erstellt simple Handlungsempfehlungen.
    """

    def recommend_for_segment(self, lookup_df: pd.DataFrame, segment: Segment, topk=3) -> list[str]:
        # sehr simple Heuristik: höchste Return_% in diesem Segment → „Aufklärung/Kaufberatung“
        seg = lookup_df[
            (lookup_df["Gender"] == segment.gender) &
            (lookup_df["Income_Label"] == segment.income_label) &
            (lookup_df["Discount_Used"] == segment.discount_used)
        ]
        if seg.empty:
            return ["Keine Daten für dieses Segment."]
        cats = seg.sort_values(["Return_%","total_purchases"], ascending=[False,False]).head(topk)["Purchase_Category"].tolist()
        return [f"Kampagne: Reduziere Returns in {c} (Segment: {segment.label()})" for c in cats]
