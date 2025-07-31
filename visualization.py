import matplotlib.pyplot as plt
import pandas as pd

def plot_interaction_overview(df: pd.DataFrame):
    # Maske: NaN = True (fehlt), False = vorhanden
    missing = df[["Social_Media_Influence", "Engagement_with_Ads"]].isna()

    # Gruppieren und zählen
    counts = missing.value_counts().reset_index()
    counts.columns = ["SM_missing", "Ads_missing", "count"]

    # Kategorien als Text definieren
    def label(row):
        if not row["SM_missing"] and not row["Ads_missing"]:
            return "Beides vorhanden"
        elif not row["SM_missing"] and row["Ads_missing"]:
            return "Nur Social Media"
        elif row["SM_missing"] and not row["Ads_missing"]:
            return "Nur Ads"
        else:
            return "Weder noch"

    counts["Kategorie"] = counts.apply(label, axis=1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(counts["Kategorie"], counts["count"], color="#6baed6")
    plt.title("Nutzerverhalten: Social Media & Ads")
    plt.xlabel("Kategorie")
    plt.ylabel("Anzahl Nutzer")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()