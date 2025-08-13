import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import patsy
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from typing import Iterable, Optional, List, Tuple
from pathlib import Path 


def get_RETURN_users(df: pd.DataFrame) -> pd.DataFrame:

    return df[
        (df["Return_Rate"] .notna())
    ]


def categorize_age(df):
    """Kategorisiert Alter in feste Gruppen mit konsistenter Reihenfolge."""
    bins = [0, 20, 30, 40, 50, 60]
    labels = ["<=20", "21–30", "31–40", "41–50", "51–60"]
    df = df.copy()
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=pd.Categorical(labels, categories=labels, ordered=True),
        right=False
    )
    return df


def categorize_income(df):
    """Income-Level zu lesbaren Labels umwandeln."""
    income_map = {1: "Low", 2: "Middle", 3: "High"}
    df = df.copy()
    df["Income_Label"] = df["Income_Level"].map(income_map)
    return df





def plot_return_heatmap(df: pd.DataFrame):
    """
    Erstellt eine Heatmap der durchschnittlichen Rückgaberate
    pro Altersgruppe und Einkaufsfrequenz.
    """
    df = categorize_age(df)

    pivot = df.pivot_table(
        values="Return_Rate",
        index="Age_Group",
        columns="Frequency_of_Purchase",
        aggfunc="mean",
        observed=False
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("🔁 Durchschnittliche Rückgaberate nach Alter & Kaufhäufigkeit")
    plt.ylabel("Altersgruppe")
    plt.xlabel("Kaufhäufigkeit")
    plt.tight_layout()
    plt.show()


def run_logistic_regression_frequency(df: pd.DataFrame):
    """
    Führt eine logistische Regression durch, um den Einfluss der
    Kaufhäufigkeit (Frequency_of_Purchase) auf die Rückgaberate zu prüfen,
    separat für jede Altersgruppe.
    """
    df = categorize_age(df)

    age_groups = df["Age_Group"].dropna().unique()

    for group in age_groups:
        print(f"\n📂 Altersgruppe: {group}")
        df_group = df[df["Age_Group"] == group].copy()

        if len(df_group) < 30:
            print(f"⚠️ Zu wenige Datenpunkte ({len(df_group)}) – übersprungen.")
            continue

        # Nur die benötigten Spalten und Drop von NAs
        df_group = df_group.dropna(subset=["Return_Rate", "Frequency_of_Purchase"])

        X = sm.add_constant(df_group["Frequency_of_Purchase"])
        y = df_group["Return_Rate"]

        try:
            model = sm.Logit(y, X).fit(disp=0)
            print(model.summary())
        except Exception as e:
            print(f"❌ Fehler bei Altersgruppe {group}: {e}")
            
            

def plot_heatmap(data, x, y, value, title):
    pivot = data.pivot_table(index=y, columns=x, values=value, aggfunc='mean', dropna=False)
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()

def heatmap_return_by_purchaseamount(df):
    df = categorize_age(df)
    df["Purchase_Amount_Bin"] = pd.qcut(df["Purchase_Amount"], 4, labels=["Low", "Mid", "High", "Very High"])
    plot_heatmap(
        df,
        x="Purchase_Amount_Bin",
        y="Age_Group",
        value="Return_Rate",
        title="🔁 Return Rate by Purchase Amount and Age Group"
    )

def heatmap_return_by_gender(df):
    df = categorize_age(df)
    df = df[df["Gender"].isin(["Male", "Female"])]
    plot_heatmap(
        df,
        x="Gender",
        y="Age_Group",
        value="Return_Rate",
        title="🔁 Return Rate by Gender and Age Group"
    )


def heatmap_return_by_income(df):
    df = categorize_age(df)

    # Income_Level explizit als Kategorie mit Labels
    income_map = {
        1: "Low",
        2: "Middle",
        3: "High"
    }
    df["Income_Label"] = df["Income_Level"].map(income_map)

    # Reihenfolge festlegen (optional, aber gut für Konsistenz)
    df["Income_Label"] = pd.Categorical(df["Income_Label"], categories=["Low", "Middle", "High"], ordered=True)

    plot_heatmap(
        df,
        x="Income_Label",
        y="Age_Group",
        value="Return_Rate",
        title="🔁 Return Rate by Income Level and Age Group"
    )




# HEATMAP GRID



def calc_return_percentage(df, group_cols):
    """
    Berechnet den Return-% pro Gruppe.
    Annahme: Return_Rate ist binär.
    """
    df = df.copy()
    grouped = df.groupby(group_cols, observed=False).agg(
        total_purchases=("Return_Rate", "count"),
        total_returns=("Return_Rate", "sum")
    ).reset_index()
    grouped["Return_%"] = (grouped["total_returns"] / grouped["total_purchases"]) * 100
    return grouped



def analyze_returns(df):
    # Alters- und Einkommenslabel berechnen
    df = categorize_age(df)
    df = categorize_income(df)

    # Nur relevante Gruppen
    df = df[df["Gender"].isin(["Male", "Female"])]
    df = df[df["Income_Label"].isin(["High", "Middle"])]

    # Definierte Kombis: 8 Heatmaps
    combos = [
        ("Female", "High", True),
        ("Female", "High", False),
        ("Female", "Middle", True),
        ("Female", "Middle", False),
        ("Male", "High", True),
        ("Male", "High", False),
        ("Male", "Middle", True),
        ("Male", "Middle", False),
    ]

    for gender, income, discount in combos:
        sub = df[
            (df["Gender"] == gender) &
            (df["Income_Label"] == income) &
            (df["Discount_Used"] == discount)
        ]

        if sub.empty:
            print(f"⚠️ Keine Daten für {gender} | {income} | Discount={discount}.")
            continue

        # % Return je Age_Group × Purchase_Category
        grouped = calc_return_percentage(sub, ["Purchase_Category", "Age_Group"])
        pivot = grouped.pivot_table(
            index="Age_Group",
            columns="Purchase_Category",
            values="Return_%",
            aggfunc="mean",
            observed=False
        )

        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm")
        plt.title(f"Return % | {gender} | {income} | Discount={discount}")
        plt.ylabel("Age Group")
        plt.xlabel("Purchase Category")
        plt.tight_layout()
        plt.show()



# REGRESSIONEN


def _build_group_df(df, gender, income_label, discount_used):
    """
    Temporäres DataFrame pro Gruppe, ohne Kategorien/Altersgruppen zu filtern.
    Original-DF bleibt unverändert.
    """
    tmp = df.copy()
    tmp = categorize_age(tmp)
    tmp = categorize_income(tmp)

    tmp = tmp[
        (tmp["Gender"] == gender) &
        (tmp["Income_Label"] == income_label) &
        (tmp["Discount_Used"] == discount_used)
    ].dropna(subset=["Return_Rate", "Purchase_Category", "Age_Group"])

    if tmp.empty or tmp["Return_Rate"].nunique() < 2:
        return pd.DataFrame()

    tmp["Return_Rate"] = tmp["Return_Rate"].astype(float)
    return tmp

def run_logit_for_group(temp_df, alpha=0.8):
    """
    Logistische Regression mit Ridge (L2):
    Return_Rate ~ C(Purchase_Category) + C(Age_Group)
    Dummy-Codierung via patsy.
    """
    y, X = patsy.dmatrices(
        "Return_Rate ~ C(Purchase_Category) + C(Age_Group)",
        data=temp_df,
        return_type="dataframe"
    )

    # Regularisierte Schätzung (stabil bei Separation/Singularitäten)
    model = sm.Logit(y, X).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)

    coef = pd.Series(model.params, index=X.columns)
    odds = np.exp(coef)
    return model, coef, odds

def logistic_regression_groupwise(
    df,
    combos=(
        ("Female", "High", True),   ("Female", "High", False),
        ("Female", "Middle", True), ("Female", "Middle", False),
        ("Male", "High", True),     ("Male", "High", False),
        ("Male", "Middle", True),   ("Male", "Middle", False),
    ),
    alpha=0.8
):
    """
    Führt pro definierter Gruppe eine logistische Regression durch:
    Return_Rate ~ Purchase_Category + Age_Group (alle Kategorien, keine Top-N).
    """
    for gender, income, disc in combos:
        sub = _build_group_df(df, gender, income, disc)
        if sub.empty:
            print(f"⚠️ Gruppe übersprungen: {gender} | {income} | Discount={disc} (zu wenig Daten/Varianz).")
            continue

        try:
            model, coef, odds = run_logit_for_group(sub, alpha=alpha)
            print(f"\n📊 Logit (Ridge) – {gender} | {income} | Discount={disc} | n={len(sub)}")
            print("Koeffizienten:")
            print(coef.round(4))
            print("\nOdds Ratios:")
            print(odds.round(3))
        except PerfectSeparationError:
            print(f"⚠️ Perfekte Separation – {gender} | {income} | Discount={disc}.")
        except np.linalg.LinAlgError as e:
            print(f"❌ Lineare Algebra-Fehler – {gender} | {income} | Discount={disc}: {e}")
        except Exception as e:
            print(f"❌ Fehler – {gender} | {income} | Discount={disc}: {e}")
    
    
    
 # ORs SAMMELN   
    
    
def collect_odds_ratios(df, alpha=0.8):
    """
    Führt die groupwise Logit-Modelle aus und sammelt Odds Ratios (OR) als Long-DF.
    Spalten: group, factor, OR
    group-Format: 'Gender | Income | Disc=True/False'
    factor-Format: 'Electronics', 'Books', ... bzw. 'Age 21–30', ...
    """
    rows = []

    combos = (
        ("Female", "High", True),   ("Female", "High", False),
        ("Female", "Middle", True), ("Female", "Middle", False),
        ("Male", "High", True),     ("Male", "High", False),
        ("Male", "Middle", True),   ("Male", "Middle", False),
    )

    for gender, income, disc in combos:
        sub = _build_group_df(df, gender, income, disc)
        if sub.empty:
            continue
        try:
            model, coef, odds = run_logit_for_group(sub, alpha=alpha)
        except Exception:
            continue

        group = f"{gender} | {income} | Disc={'True' if disc else 'False'}"

        for name, orv in odds.items():
            # Intercepts weglassen
            if name.lower() in ("const", "intercept"):
                continue

            # Namen wie 'C(Purchase_Category)[T.Electronics]' -> 'Electronics'
            # und 'C(Age_Group)[T.31–40]' -> 'Age 31–40'
            factor = None
            if name.startswith("C(Purchase_Category)"):
                try:
                    factor = name.split("T.", 1)[1].rstrip("]")
                except Exception:
                    continue
            elif name.startswith("C(Age_Group)"):
                try:
                    lvl = name.split("T.", 1)[1].rstrip("]")
                    factor = f"Age {lvl}"
                except Exception:
                    continue

            if factor is not None:
                rows.append({"group": group, "factor": factor, "OR": float(orv)})

    return pd.DataFrame(rows)




# SEGMENT-RETURN-DATENBANK:



def build_return_rate_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt eine Lookup-Tabelle mit Return-% pro Segment, Altersgruppe & Produkt.
    Spalten: Gender, Income_Label, Discount_Used, Age_Group, Purchase_Category,
             total_purchases, total_returns, Return_%
    Annahme: Return_Rate ist binär (0/1).
    """
    df = categorize_age(df)
    df = categorize_income(df)

    grouped = df.groupby(
        ["Gender", "Income_Label", "Discount_Used", "Age_Group", "Purchase_Category"],
        observed=False
    ).agg(
        total_purchases=("Return_Rate", "count"),
        total_returns=("Return_Rate", "sum")
    ).reset_index()

    grouped["Return_%"] = (grouped["total_returns"] / grouped["total_purchases"]) * 100

    # Für robuste Serialisierung: Kategorien -> String (optional, aber praktisch)
    grouped["Age_Group"] = grouped["Age_Group"].astype(str)
    grouped["Income_Label"] = grouped["Income_Label"].astype(str)
    grouped["Gender"] = grouped["Gender"].astype(str)
    grouped["Purchase_Category"] = grouped["Purchase_Category"].astype(str)

    return grouped


# ---------------- Speichern & Laden ------------------------------------------

def save_return_rate_lookup(lookup_df: pd.DataFrame, path: str) -> None:
    """
    Speichert die Lookup-Tabelle immer als CSV.
    Legt fehlende Verzeichnisse automatisch an.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Erzwingt CSV-Endung
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")

    lookup_df.to_csv(p, index=False)


def load_return_rate_lookup(path: str) -> pd.DataFrame:
    """
    Lädt die Lookup-Tabelle als CSV.
    """
    if not path.lower().endswith(".csv"):
        raise ValueError("Bitte eine .csv-Datei angeben.")
    return pd.read_csv(path)


# ---------------- Abfragen & Sichten -----------------------------------------

def filter_lookup(
    lookup_df: pd.DataFrame,
    gender: Optional[str] = None,
    income_label: Optional[str] = None,
    discount_used: Optional[bool] = None,
    age_groups: Optional[Iterable[str]] = None,
    categories: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Flexibler Filter auf der Lookup-Tabelle ohne Mindestanzahl an Käufen.
    """
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


def get_topk_categories(
    lookup_df: pd.DataFrame,
    gender: str,
    income_label: str,
    discount_used: bool,
    age_group: str,
    k: int = 5,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Liefert die Top-K Produktkategorien nach Return_% für ein genaues Segment inkl. Age_Group.
    Tie-Breaker: total_purchases (absteigend).
    """
    seg = filter_lookup(
        lookup_df,
        gender=gender,
        income_label=income_label,
        discount_used=discount_used,
        age_groups=[age_group]
    )

    if seg.empty:
        return seg

    seg = seg.copy()
    out = seg.sort_values(
        by=["Return_%", "total_purchases"],
        ascending=[ascending, False]
    ).head(k)

    return out[
        ["Gender", "Income_Label", "Discount_Used", "Age_Group", "Purchase_Category",
         "total_purchases", "total_returns", "Return_%"]
    ].reset_index(drop=True)


def pivot_segment_age_product(
    lookup_df: pd.DataFrame,
    gender: str,
    income_label: str,
    discount_used: bool
) -> pd.DataFrame:
    """
    Für ein Segment (Gender, Income, Discount) eine Pivot-Tabelle:
    Zeilen = Age_Group, Spalten = Purchase_Category, Werte = Return_%.
    """
    seg = filter_lookup(
        lookup_df,
        gender=gender,
        income_label=income_label,
        discount_used=discount_used
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


def pivot_with_counts(
    lookup_df: pd.DataFrame,
    gender: str,
    income_label: str,
    discount_used: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wie pivot_segment_age_product, aber zusätzlich ein Pivot der total_purchases (Counts).
    Rückgabe: (pivot_value(Return_%), pivot_count(total_purchases_sum))
    """
    seg = filter_lookup(
        lookup_df,
        gender=gender,
        income_label=income_label,
        discount_used=discount_used
    )
    if seg.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot_val = seg.pivot_table(
        index="Age_Group",
        columns="Purchase_Category",
        values="Return_%",
        aggfunc="mean",
        observed=False
    )
    pivot_n = seg.pivot_table(
        index="Age_Group",
        columns="Purchase_Category",
        values="total_purchases",
        aggfunc="sum",
        observed=False
    )
    return pivot_val, pivot_n


def build_heatmap_annotations(
    pivot_val: pd.DataFrame,
    pivot_n: pd.DataFrame,
    low_base_threshold: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baut ein Annotation-DataFrame für Heatmaps: z.B. '80% (n=5)' pro Zelle.
    Zusätzlich wird eine Maske (True/False) für 'kleine Basis' (n < threshold) geliefert.
    """
    # Index/Kolumnen harmonisieren
    pivot_n = pivot_n.reindex_like(pivot_val)

    annot = pivot_val.copy().astype(object)
    low_base_mask = pd.DataFrame(False, index=pivot_val.index, columns=pivot_val.columns)

    for i in pivot_val.index:
        for j in pivot_val.columns:
            v = pivot_val.loc[i, j]
            n = pivot_n.loc[i, j] if (i in pivot_n.index and j in pivot_n.columns) else np.nan
            if pd.isna(v):
                annot.loc[i, j] = ""
            else:
                n_int = int(n) if pd.notna(n) else 0
                annot.loc[i, j] = f"{v:.0f}% (n={n_int})"
                if pd.notna(n) and n < low_base_threshold:
                    low_base_mask.loc[i, j] = True

    return annot, low_base_mask


def summarize_segments(
    lookup_df: pd.DataFrame,
    combos: Optional[List[Tuple[str, str, bool]]] = None
) -> pd.DataFrame:
    """
    Kompakte Zusammenfassung je Segment + Age_Group (ohne Mindestanzahl-Filter):
    - Zeilen: Gender, Income_Label, Discount_Used, Age_Group
    - Kennzahlen: base_purchases (Summe Käufe), n_categories (Anzahl Kat. mit Daten),
                  mean_Return_%, max_Return_%, min_Return_%
    """
    if combos is None:
        combos = [
            ("Female", "High", True),   ("Female", "High", False),
            ("Female", "Middle", True), ("Female", "Middle", False),
            ("Male", "High", True),     ("Male", "High", False),
            ("Male", "Middle", True),   ("Male", "Middle", False),
        ]

    rows: List[pd.DataFrame] = []

    for gender, income, disc in combos:
        seg = filter_lookup(
            lookup_df,
            gender=gender,
            income_label=income,
            discount_used=disc
        )
        if seg.empty:
            continue

        # Named Aggregations (robust) inkl. base_purchases
        agg = (
            seg.groupby(["Gender", "Income_Label", "Discount_Used", "Age_Group"], observed=False)
              .agg(**{
                  "base_purchases": ("total_purchases", "sum"),
                  "n_categories": ("Return_%", "count"),
                  "mean_Return_%": ("Return_%", "mean"),
                  "max_Return_%": ("Return_%", "max"),
                  "min_Return_%": ("Return_%", "min"),
              })
              .reset_index()
        )
        rows.append(agg)

    if not rows:
        return pd.DataFrame(
            columns=[
                "Gender", "Income_Label", "Discount_Used", "Age_Group",
                "base_purchases", "n_categories", "mean_Return_%",
                "max_Return_%", "min_Return_%"
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    cols = [
        "Gender", "Income_Label", "Discount_Used", "Age_Group",
        "base_purchases", "n_categories", "mean_Return_%",
        "max_Return_%", "min_Return_%"
    ]
    out = out[cols].sort_values(
        by=["Gender", "Income_Label", "Discount_Used", "Age_Group"]
    ).reset_index(drop=True)

    return out
