import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


# -----------------------------------------------------------------------------
# 1) Social/Ads – Datenverfügbarkeit (wie in deiner alten visualisation.py)
# -----------------------------------------------------------------------------
def _used_from_category(s: pd.Series) -> pd.Series:
    """
    Interpretiert kategoriale Nutzung:
      - True  : High / Medium / Low
      - False : None  (oder fehlend)
    Robust: case-insensitive, trimmt Whitespace.
    """
    if s is None:
        return pd.Series(dtype=bool)
    v = s.astype(str).str.strip().str.lower()
    return v.isin({"high", "medium", "low"})

def plot_social_ads_fourbars(
    df: pd.DataFrame,
    sm_col: str = "Social_Media_Influence",
    ads_col: str = "Engagement_with_Ads",
    title: str = "Nutzer nach Nutzung von Social Media / Ads (kategorial)"
) -> None:
    """
    Zeigt genau 4 Balken (Anzahl Nutzer) für kategoriale Spalten mit Levels:
      {High, Medium, Low, None}.
    'Genutzt' = High/Medium/Low, 'Nicht genutzt' = None/fehlend.
    """
    if sm_col not in df.columns or ads_col not in df.columns:
        raise KeyError(f"Spalten '{sm_col}' und/oder '{ads_col}' fehlen im DataFrame.")

    sm_used  = _used_from_category(df[sm_col])
    ads_used = _used_from_category(df[ads_col])

    both     = int((sm_used & ads_used).sum())
    only_sm  = int((sm_used & ~ads_used).sum())
    only_ads = int((~sm_used & ads_used).sum())
    neither  = int((~sm_used & ~ads_used).sum())

    cats   = ["Beides", "Nur Social Media", "Nur Ads", "Weder noch"]
    counts = [both, only_sm, only_ads, neither]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(cats, counts)
    for bar, val in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val),
                 ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.xlabel("Kategorie")
    plt.ylabel("Anzahl Nutzer")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 2) Logit (Ridge) – OR-Heatmap (Farbe: log2(OR), Zahl: OR) mit rotierten Zahlen
#    Erwartet df_or aus ReturnModeler.collect_odds_ratios():
#    Spalten: group, factor, OR
# -----------------------------------------------------------------------------
def plot_return_logit_summary(df_or: pd.DataFrame) -> None:
    if df_or is None or df_or.empty:
        print("⚠️ Keine Odds-Ratio-Daten übergeben.")
        return

    # Pivot: Zeilen = Gruppen, Spalten = Faktoren, Werte = OR
    M_or = df_or.pivot_table(index="group", columns="factor", values="OR", aggfunc="mean")

    # Reihenfolge (optional, falls vorhanden)
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

    # Numerik + Validierungen
    M_or = M_or.apply(pd.to_numeric, errors="coerce")
    M_or = M_or.replace([np.inf, -np.inf], np.nan)
    M_or = M_or.where(M_or > 0, np.nan)  # OR muss >0 sein

    # Farbe über log2(OR), sanft clippen
    M_log2 = np.log2(M_or)
    vmin, vmax = -2.0, 2.0  # grob OR in [0.25, 4]
    M_plot = M_log2.clip(lower=vmin, upper=vmax)

    # Annotation: tatsächliche OR (gerundet), leere Felder ohne Text
    annot = M_or.round(2).astype(object).where(~M_or.isna(), "")

    if M_plot.empty or M_plot.shape[1] == 0:
        print("⚠️ Nichts zu plotten (leere Matrix nach Vorbereitung).")
        return

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

    for t in ax.texts:
        if t.get_text():
            t.set_ha("center")
            t.set_va("center")

    ax.set_title("Return Drivers (Logit, Ridge) – Odds Ratios pro Gruppe\nFarbe: log2(OR) • Zahl: OR")
    ax.set_xlabel("Faktoren (Produkte & Altersgruppen)")
    ax.set_ylabel("Gruppen")
    plt.xticks(rotation=35, ha="right")

    plt.subplots_adjust(left=0.18, right=0.96, bottom=0.24, top=0.90)
    plt.show()


# -----------------------------------------------------------------------------
# 3) Segment-Heatmap (Return_% + Counts) – nutzt die Services-Helfer
#    aus services.returns (pivot_with_counts & build_heatmap_annotations)
# -----------------------------------------------------------------------------
def plot_segment_heatmap(
    lookup_df: pd.DataFrame,
    gender: str,
    income_label: str,
    discount_used: bool,
    low_base_threshold: int = 3,
    fade_low_base: bool = True,
) -> None:
    """
    Heatmap für ein Segment (Gender, Income, Discount) mit Annotationen "XX% (n=YY)".
    Kleine Basen (n < low_base_threshold) werden je nach Einstellung
      - fade_low_base=True: halbtransparent grau überlagert
      - fade_low_base=False: per Mask ausgeblendet

    Erwartet eine Lookup-Tabelle aus ReturnLookupService.build_lookup().
    """
    # Lazy-Import, um zirkuläre Importe zu vermeiden
    from services.returns import ReturnLookupService as RLS

    pivot_val, pivot_n = RLS.pivot_with_counts(
        lookup_df,
        gender=gender,
        income_label=income_label,
        discount_used=discount_used,
    )

    if pivot_val.empty:
        print(f"⚠️ Keine Daten für Segment: {gender} | {income_label} | Disc={discount_used}")
        return

    annot, low_mask = RLS.build_heatmap_annotations(
        pivot_val, pivot_n, low_base_threshold=low_base_threshold
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Haupt-Heatmap (Werte sind Return_%)
    sns.heatmap(
        pivot_val,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=50,  # 50% als neutraler Mittelpunkt
        cbar_kws={"label": "Return %"},
        ax=ax,
    )

    # Option 1: Kleine Basen halbtransparent abdunkeln (Overlay)
    if fade_low_base and low_mask.values.any():
        overlay = low_mask.astype(int).reindex_like(pivot_val).values
        ax.imshow(
            overlay,
            cmap="Greys",
            alpha=np.where(overlay == 1, 0.35, 0.0),
            extent=(0, overlay.shape[1], overlay.shape[0], 0),
            aspect="auto",
            interpolation="nearest",
        )
    # Option 2: Kleine Basen komplett maskieren
    elif not fade_low_base:
        plt.cla()
        sns.heatmap(
            pivot_val,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            center=50,
            mask=low_mask,
            cbar_kws={"label": "Return %"},
            ax=ax,
        )

    # <-- Rotation IMMER nach dem Zeichnen (egal welcher Zweig)
    for t in ax.texts:
        if t.get_text():
            t.set_rotation(90)
            t.set_ha("center")
            t.set_va("center")

    ax.set_title(f"Return % by Age Group & Product Category\n({gender} | {income_label} | Discount={discount_used})")
    ax.set_xlabel("Purchase Category")
    ax.set_ylabel("Age Group")
    plt.xticks(rotation=35, ha="right")

    plt.subplots_adjust(left=0.18, right=0.96, bottom=0.22, top=0.90)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os
    import sys

    # Damit der services-Ordner gefunden wird
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from services.repository import DataRepository
    from services.modeling import ReturnModeler
    from services.returns import ReturnLookupService

    # CSV laden
    repo = DataRepository("customer_behavour/Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = repo.load_and_prepare()

    # 1) Social/Ads – Vier Balken
    plot_social_ads_fourbars(df)

    # 2) Logit-OR-Heatmap
    rm = ReturnModeler(df)
    df_or = rm.collect_odds_ratios(alpha=0.8, allow_single_class=True)
    plot_return_logit_summary(df_or)

    # 3) Segment-Heatmap für Beispielsegment
    rl = ReturnLookupService(df)
    lookup = rl.build_lookup()
    plot_segment_heatmap(
        lookup_df=lookup,
        gender="Female",
        income_label="High",
        discount_used=True
    )