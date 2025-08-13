import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_interaction_overview(df: pd.DataFrame):
    # Neue Logik: 0 = fehlt, >0 = vorhanden
    def map_presence(value):
        return value == 0  # True = fehlt, False = vorhanden

    missing = pd.DataFrame({
        "SM_missing": df["Social_Media_Influence"].apply(map_presence),
        "Ads_missing": df["Engagement_with_Ads"].apply(map_presence)
    })

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

    # Plot 1
    plt.figure(figsize=(8, 6))
    plt.bar(counts["Kategorie"], counts["count"], color="#6baed6")
    plt.title("Nutzerverhalten: Social Media & Ads (Datenverfügbarkeit)")
    plt.xlabel("Kategorie")
    plt.ylabel("Anzahl Nutzer")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_return_logit_summary(or_df: pd.DataFrame):
    """
    Heatmap echter Odds Ratios (OR):
    - Farbe: log2(OR) (0 = neutral / OR=1)
    - Annotation: OR gerundet
    Erwartet Long-DF: columns={group, factor, OR}
    """
    if or_df is None or or_df.empty:
        print("⚠️ Keine Odds-Ratio-Daten übergeben.")
        return

    # Pivot
    M_or = or_df.pivot_table(index="group", columns="factor", values="OR", aggfunc="mean")

    # Reihenfolge (optional)
    group_order = [
        "Female | High | Disc=True",
        "Female | High | Disc=False",
        "Female | Middle | Disc=True",
        "Female | Middle | Disc=False",
        "Male | High | Disc=True",
        "Male | High | Disc=False",
        "Male | Middle | Disc=True",
        "Male | Middle | Disc=False",
    ]
    M_or = M_or.reindex(index=[g for g in group_order if g in M_or.index])

    # Sicherstellen: numerisch
    M_or = M_or.apply(pd.to_numeric, errors="coerce")

    # OR <= 0 oder ±inf → NaN (log2 nicht definiert)
    M_or = M_or.replace([np.inf, -np.inf], np.nan)
    M_or = M_or.where(M_or > 0, np.nan)

    # Farbe: log2(OR), dann clippen
    M_log2 = np.log2(M_or).replace([np.inf, -np.inf], np.nan)
    vmin, vmax = -2.0, 2.0  # ≈ OR in [0.25, 4]
    M_plot = M_log2.clip(lower=vmin, upper=vmax)

    # Annotationen: OR gerundet, fehlende als ""
    annot = M_or.round(2).astype(object).where(~M_or.isna(), "")

    # Nichts zu plotten?
    if M_plot.empty or M_plot.shape[1] == 0:
        print("⚠️ Nichts zu plotten (leere Matrix nach Vorbereitung).")
        return

    # Plot 2 (Heatmap)
    plt.figure(figsize=(max(12, 0.6 * len(M_plot.columns) + 6), 6.5))
    ax = sns.heatmap(
        M_plot.astype(float),
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        annot=annot,
        fmt="",
        cbar_kws={"label": "log2(OR)  (0 = neutral, + = höheres Return-Risiko)"},
        linewidths=0.5,
        linecolor="white",
    )

    # Annotationen im Feld um 90° drehen (nur nicht-leere)
    for t in ax.texts:
        if t.get_text():
            t.set_ha("center")
            t.set_va("center")

    ax.set_title("Return Drivers (Logit, Ridge) – Odds Ratios pro Gruppe\nFarbe: log2(OR) • Zahl: OR")
    ax.set_xlabel("Faktoren (Produkte & Altersgruppen)")
    ax.set_ylabel("Gruppen")
    plt.xticks(rotation=35, ha="right")

    # Mehr Platz links & rechts kleiner, unten etwas größer
    plt.subplots_adjust(left=0.18, right=0.96, bottom=0.24, top=0.90)

    plt.show()


# ---------------------------------------------------------------------------
# Segment-Heatmap (Return_% + Counts) – nutzt die Lookup-Tabelle und Helpers
# aus Return.py: pivot_with_counts und build_heatmap_annotations
# ---------------------------------------------------------------------------

def plot_segment_heatmap(
    lookup_df: pd.DataFrame,
    gender: str,
    income_label: str,
    discount_used: bool,
    low_base_threshold: int = 3,
    fade_low_base: bool = True
) -> None:
    """
    Heatmap für ein Segment (Gender, Income, Discount) mit Annotationen "XX% (n=YY)".
    Kleine Basen (n < low_base_threshold) werden je nach Einstellung
    - fade_low_base=True: halbtransparent grau überlagert
    - fade_low_base=False: per Mask ausgeblendet

    Erwartet, dass in Return.py vorhanden sind:
      - pivot_with_counts(lookup_df, gender, income_label, discount_used)
      - build_heatmap_annotations(pivot_val, pivot_n, low_base_threshold)
    """
    # Lazy import, um zirkuläre Importe zu vermeiden
    from Return import pivot_with_counts, build_heatmap_annotations

    pivot_val, pivot_n = pivot_with_counts(
        lookup_df,
        gender=gender,
        income_label=income_label,
        discount_used=discount_used
    )

    if pivot_val.empty:
        print("⚠️ Keine Daten für dieses Segment.")
        return

    annot, low_mask = build_heatmap_annotations(
        pivot_val,
        pivot_n,
        low_base_threshold=low_base_threshold
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    # Haupt-Heatmap
    sns.heatmap(
        pivot_val,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=50,
        cbar_kws={'label': 'Return %'},
        ax=ax
    )

    # Annotationen im Feld um 90° drehen (nur nicht-leere)
    for t in ax.texts:
        if t.get_text():
            t.set_rotation(90)
            t.set_ha("center")
            t.set_va("center")
            
            
    # Option 1: Kleine Basen halbtransparent abdunkeln (Overlay)
    if fade_low_base and low_mask.values.any():
        overlay = low_mask.astype(int).reindex_like(pivot_val).values
        ax.imshow(
            overlay,
            cmap="Greys",
            alpha=np.where(overlay == 1, 0.35, 0.0),
            extent=(0, overlay.shape[1], overlay.shape[0], 0),
            aspect="auto",
            interpolation="nearest"
        )
    # Option 2: Kleine Basen einfach ausblenden
    elif not fade_low_base:
        plt.cla()  # Achse leeren und mit Maske neu zeichnen
        sns.heatmap(
            pivot_val,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            center=50,
            mask=low_mask,
            cbar_kws={'label': 'Return %'},
            ax=ax
        )

    ax.set_title(f"Return % by Age Group & Product Category\n({gender} | {income_label} | Discount={discount_used})")
    ax.set_xlabel("Purchase Category")
    ax.set_ylabel("Age Group")

    # Labels etwas lesbarer machen
    plt.xticks(rotation=35, ha="right")

    # Platzränder: links/rechts kompakter, unten etwas Raum für xticks
    plt.subplots_adjust(left=0.18, right=0.96, bottom=0.22, top=0.90)

    plt.tight_layout()
    plt.show()
