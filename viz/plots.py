import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import textwrap
import re


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



# -----------------------------------------------------------------------------
# 4) Return-Tabelle 
#    Top 29 Werte Segmentübergreifend (nach Return_% & total_purchases)
# -----------------------------------------------------------------------------



def plot_top_returns_table(df, top_n=20, title="Top Returns – Kategorie & Segment-übergreifend"):
    """
    Zeigt die top_n Zeilen mit den höchsten Return_% als Tabelle (Matplotlib).
    Erwartet ein DataFrame wie lookup_clean.
    Zeilen ohne Discount_Used werden leicht grau hinterlegt.
    """
    import matplotlib.pyplot as plt

    print("[plot_top_returns_table] ACTIVE v3 — bbox/scale tuning enabled")

    # sortieren
    top_df = df.sort_values(
        by=["Return_%", "total_purchases"],
        ascending=[False, False]
    ).head(top_n).reset_index(drop=True)

    # --- adaptive Spaltenbreiten nach Textlänge ---
    cols = list(top_df.columns)
    # Länge pro Spalte abschätzen (Header + Werte)
    lengths = []
    for c in cols:
        vals = top_df[c].astype(str).tolist()
        max_len = max([len(c)] + [len(v) for v in vals])
        lengths.append(max_len)

    # Kategorie bekommt zusätzlich einen Bonus (sichtbarer breiter)
    try:
        cat_idx = cols.index("Purchase_Category")
        lengths[cat_idx] = int(lengths[cat_idx] * 1.6)
    except ValueError:
        pass

    # normieren auf 1.0
    total = float(sum(lengths)) if sum(lengths) > 0 else 1.0
    col_widths = [L / total for L in lengths]

    # --- Figure & Achse ---
    fig, ax = plt.subplots(figsize=(14, min(0.30 * len(top_df) + 2.2, 15)))
    # möglichst volle Fläche nutzen
    ax.set_position([0.01, 0.02, 0.98, 0.92])  # [left, bottom, width, height]
    ax.axis("off")

    # --- Tabelle rendern — fast volle Fläche, nahe am Titel ---
    table = ax.table(
        cellText=top_df.values,
        colLabels=cols,
        colWidths=col_widths,      # adaptive Breiten
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],   # maximal ausnutzen
    )

    # Schrift & Zeilenhöhe kompakt
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 0.6)  # schmalere Zeilen

    # Zeilen ohne Discount leicht grau einfärben
    if "Discount_Used" in top_df.columns:
        discount_col_idx = cols.index("Discount_Used")
        for i in range(len(top_df)):
            if not bool(top_df.iloc[i, discount_col_idx]):
                for j in range(len(cols)):
                    table[(i + 1, j)].set_facecolor("#f0f0f0")  # +1 wegen Headerzeile

    # Titel sehr nah an Tabelle
    ax.set_title(title, fontsize=14, pad=8)

    # KEIN tight_layout, um bbox nicht zu zerstören
    plt.show()
    
    
    
    
# -----------------------------------------------------------------------------
# 5) RECOMMENDATIONS 
#    Je top 3 pro Altersgruppe (über ALLE Segmente)
# -----------------------------------------------------------------------------
    
    
    
    
def plot_recommendations_topk_all_ages_table(
    reco_df: pd.DataFrame,
    k: int = 3,
    title: str = "Top 3 je Altersgruppe – nach Käufen & Rückgabe"
) -> None:
    """
    Eine einzige Tabelle: kombiniere pro Altersgruppe die Top-k Zeilen.
    Ranking pro Altersgruppe: total_purchases DESC, dann Return_% DESC.
    Altersreihenfolge: <=20, 21–30, 31–40, 41–50, 51–60.
    Recommendation-Spalte ist doppelt so breit.
    """
    if reco_df is None or reco_df.empty:
        print("⚠️ Keine Empfehlungstabellen zu plotten.")
        return

    age_order = ["<=20","21–30","31–40","41–50","51–60"]
    frames = []
    for age in age_order:
        sub = (
            reco_df[reco_df["Age_Group"].astype(str) == age]
            .sort_values(["total_purchases","Return_%"], ascending=[False, False])
            .head(k)
            .reset_index(drop=True)
        )
        if not sub.empty:
            frames.append(sub.assign(Age_Group=age))

    if not frames:
        print("⚠️ Keine Daten in den ausgewählten Altersgruppen.")
        return

    combined = pd.concat(frames, ignore_index=True)

    view = combined[
        [
            "Age_Group",
            "Segment",
            "Purchase_Category",
            "total_purchases",
            "Return_%",
            "Ads_Status",
            "Loyalty_Status",
            "Recommendation",
        ]
    ].copy()

    # Recommendation formattieren:
    # - genau 2 Bausteine -> harter Zeilenumbruch zwischen beiden
    # - sonst normaler Wrap
    wrap_width = 70
    def _fmt_reco(txt: str) -> str:
    # Split an jedem '|' mit beliebigem Whitespace drum herum
        parts = [p.strip() for p in re.split(r"\s*\|\s*", str(txt)) if p.strip()]
        if len(parts) > 1:
            # Immer harter Zeilenumbruch zwischen den Bausteinen
            return "\n".join(parts)
        # Falls nur ein Teil, ganz normal umbrechen
        return "\n".join(textwrap.wrap(parts[0], width=wrap_width))
    
    view["Recommendation"] = view["Recommendation"].apply(_fmt_reco)

    # Figure und Tabellen-Layout
    fig_height = min(1 + 0.55 * len(view), 28)
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.axis("off")

    # Recommendation doppelt so breit
    col_widths = [1.0, 1.2, 1.6, 1.0, 1.0, 1.1, 1.1, 2.2]

    tbl = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        colWidths=col_widths,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)

    # Grau hinterlegen, wenn Discount_Used == False
    if "Discount_Used" in combined.columns:
        for i, used in enumerate(combined["Discount_Used"].astype(bool).tolist()):
            if not used:
                for j in range(view.shape[1]):
                    tbl[(i + 1, j)].set_facecolor("#f5f5f5")

    plt.title(title, fontsize=16, pad=12)
    plt.tight_layout()
    plt.show()