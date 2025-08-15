from services.repository import DataRepository 
from services.returns import ReturnLookupService
from services.modeling import ReturnModeler
from domain.segment import Segment
from viz.plots import (plot_social_ads_fourbars, 
                        plot_return_logit_summary, 
                        plot_segment_heatmap, 
                        plot_top_returns_table, 
                        plot_recommendations_topk_all_ages_table)
from services.rules import RuleEngine
from services.reco import CampaignRecommender
import pandas as pd



def main():
    # 1) Daten laden & vorbereiten
    repo = DataRepository("customer_behavour/Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    dfp = repo.load_and_prepare()

    # 1a) Social/Ads-Plot
    plot_social_ads_fourbars(
        dfp,
        sm_col="Social_Media_Influence",
        ads_col="Engagement_with_Ads",
        title="Social Media vs Ads Nutzung"
    )

    # 2) Lookup bauen & speichern
    rl = ReturnLookupService(dfp)
    lookup_raw = rl.build_lookup()
    rl.save_csv(lookup_raw, "artifacts/return_rate_lookup.csv")

    # 3) Regeln anwenden (Basen markieren/filtern, High-Risk finden)
    rules = RuleEngine(min_base=3, cap_or=8.0)
    lookup_flagged = rules.mark_low_base(lookup_raw)
    lookup_clean = rules.filter_low_base(lookup_flagged)
    high_risk = rules.flag_high_risk(lookup_flagged, min_return_pct=80.0)

    # optional: speichern/anzeigen -> verhindert "unused variable"
    rl.save_csv(lookup_clean, "artifacts/return_rate_lookup_clean.csv")
    rl.save_csv(high_risk,    "artifacts/high_risk_cells.csv")
    print(f"\n🔎 Lookup clean: {len(lookup_clean)} Zeilen | High-Risk: {len(high_risk)} Zeilen")
    if not high_risk.empty:
        print(high_risk.head())

    # 4) Globale Top-5 auf BASIS DER GEREINIGTEN TABELLE
    top5_global = rl.global_topk(lookup_clean, k=5)
    print("\n🌍 Globale Top-5 (höchste Return_% & viele Käufe, low_base entfernt):")
    print(top5_global)

    # 5) OR-Heatmap (Logit Ridge) – ORs clippen, DANN plotten
    modeler = ReturnModeler(dfp)
    or_df = modeler.collect_odds_ratios(alpha=0.8)
    or_df = rules.clip_odds(or_df)
    plot_return_logit_summary(or_df)

    # 6) Segment-Heatmap inkl. „n“ Overlay — mit lookup_clean plotten
    seg = Segment("Female", "High", True)
    plot_segment_heatmap(
        lookup_df=lookup_clean,
        gender=seg.gender,
        income_label=seg.income_label,
        discount_used=seg.discount_used,
        low_base_threshold=3,
        fade_low_base=True
    )
    
    plot_top_returns_table(lookup_clean, top_n=20)
    
    
    
    # 7) Kampagnenempfehlungen für ein Segment
    reco = CampaignRecommender(use_high_risk=True)
    reco_all = reco.build_recommendations_all(
        lookup_df=lookup_clean,
        raw_df=dfp,
        return_pct_min=51.0
    )
    
    # 2) ALLE Segmente sind enthalten
# 3) Sortierung für CSV: nach Alter (geordnet), dann total_purchases DESC, Return_% DESC
    age_order = ["<=20","21–30","31–40","41–50","51–60"]
    reco_all["Age_Group"] = pd.Categorical(reco_all["Age_Group"], categories=age_order, ordered=True)
    reco_all_sorted = reco_all.sort_values(
        ["Age_Group","total_purchases","Return_%"], ascending=[True, False, False]
    ).reset_index(drop=True)

    reco_csv = "artifacts/recommendations_all_segments_over51.csv"
    reco_all_sorted.to_csv(reco_csv, index=False)
    print(f"💾 Empfehlungen gespeichert: {reco_csv} | Zeilen: {len(reco_all_sorted)}")

    # Anzeige: eine große Tabelle mit Top 5 pro Altersgruppe (über ALLE Segmente)
    plot_recommendations_topk_all_ages_table(reco_all_sorted, k=3)
    



if __name__ == "__main__":
    main() 
