
from services.repository import DataRepository
from services.returns import ReturnLookupService
from services.modeling import ReturnModeler
from domain.segment import Segment
from visualisation import plot_return_logit_summary, plot_segment_heatmap


def main():
    # 1) Daten laden & vorbereiten
    repo = DataRepository("customer_behavour/Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    dfp = repo.load_and_prepare()

    # 2) Lookup bauen & speichern
    rl = ReturnLookupService(dfp)
    lookup = rl.build_lookup()
    rl.save_csv(lookup, "artifacts/return_rate_lookup.csv")

    # 3) Globale Top‑5 (nur Male/Female), sortiert nach Return_% DESC, total_purchases DESC
    top5_global = rl.global_topk(lookup, k=5)
    print("\n🌍 Globale Top‑5 (höchste Return_% & viele Käufe):")
    print(top5_global)

    # 4) OR-Heatmap (Logit Ridge)
    modeler = ReturnModeler(dfp)
    or_df = modeler.collect_odds_ratios(alpha=0.8)
    plot_return_logit_summary(or_df)

    # 5) Segment-Heatmap inkl. „n“ Overlay
    seg = Segment("Female","High",True)
    plot_segment_heatmap(
        lookup_df=lookup,
        gender=seg.gender,
        income_label=seg.income_label,
        discount_used=seg.discount_used,
        low_base_threshold=3,
        fade_low_base=True
    )

if __name__ == "__main__":
    main()
