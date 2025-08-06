import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_ads_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gibt Nutzer:innen zurück, die als "Ads-User" zählen:
    - tatsächliche Interaktion mit Ads (Engagement_with_Ads > 0)
    - UND Social Media Influence > 0 (nicht NaN)
    """
    return df[
        (df["Engagement_with_Ads"] > 0) &
        (df["Social_Media_Influence"].notna()) &
        (df["Social_Media_Influence"] > 0)
    ]


def describe_ads_users(df: pd.DataFrame):
    ads_users = get_ads_users(df)
    print("\n📊 Anzahl Ads-Nutzer (Engagement > 0):", len(ads_users))
    print("\n📈 Verteilung der Engagement-Werte:")
    print(df["Engagement_with_Ads"].value_counts().sort_index())
    
    print("\n🔍 Vorschau:")

def plot_ads_engagement_distribution(df: pd.DataFrame):
    """Visualisiert die Verteilung der Engagement-Werte (0–3) mit Ads."""
    counts = df["Engagement_with_Ads"].value_counts().sort_index()

    # Optional: Mapping zu Labels
    labels = {
        0: "Kein Interesse",
        1: "Niedrig",
        2: "Mittel",
        3: "Hoch"
    }
    x_labels = [labels.get(i, str(i)) for i in counts.index]

    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, counts.values, color="#ff7f0e")
    plt.title("Verteilung des Add-Engagement-Levels")
    plt.xlabel("Engagement-Level")
    plt.ylabel("Anzahl Nutzer")
    plt.tight_layout()
    plt.show()


def barplots_vs_ads(df: pd.DataFrame):
    target_columns = [
        "Discount_Used",
        "Customer_Satisfaction",
        "Return_Rate",
        "Product_Rating",
        "Purchase_Amount",
        "Brand_Loyalty",
        "Frequency_of_Purchase",
        "Purchase_Intent"
    ]

    for col in target_columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x="Engagement_with_Ads", y=col, estimator='mean', errorbar=None)
        plt.title(f"{col} vs Engagement_with_Ads")
        plt.ylabel("Durchschnitt")
        plt.xlabel("Engagement mit Ads (0–3)")
        plt.tight_layout()
        plt.show()


def barplots_intent_vs_targets(df: pd.DataFrame):
    """Zeigt Durchschnittswerte für Return Rate & Satisfaction nach Kaufabsicht (nur Ads-Nutzer)."""
    ads_users = get_ads_users(df)

    target_columns = ["Return_Rate", "Customer_Satisfaction"]

    for col in target_columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=ads_users, x="Purchase_Intent", y=col, estimator="mean", errorbar=None)
        plt.title(f"{col} vs Purchase Intent (nur Ads-Nutzer)")
        plt.xlabel("Kaufabsicht (0–3)")
        plt.ylabel("Durchschnitt")
        plt.tight_layout()
        plt.show()


def barplots_discount_vs_targets(df: pd.DataFrame):
    """Zeigt Durchschnittswerte für Return Rate & Satisfaction nach Discount-Nutzung (nur Ads-Nutzer)."""
    ads_users = get_ads_users(df)

    target_columns = ["Customer_Satisfaction", "Return_Rate"]

    for col in target_columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=ads_users, x="Discount_Used", y=col, estimator="mean", errorbar=None)
        plt.title(f"{col} vs Discount Used (nur Ads-Nutzer)")
        plt.xlabel("Discount genutzt (0 = nein, 1 = ja)")
        plt.ylabel("Durchschnitt")
        plt.tight_layout()
        plt.show()


def heatmap_return_by_category_and_ads(df: pd.DataFrame):
    """Heatmap: Rückgabequote je Produktkategorie und Ads-Engagement."""
    ads_users = get_ads_users(df)

    # Gruppieren: mittlere Rückgabequote pro Kategorie und Engagement-Level
    pivot = ads_users.pivot_table(
        index="Purchase_Category",
        columns="Engagement_with_Ads",
        values="Return_Rate",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5)
    plt.title("📦 Rückgabequote nach Produktkategorie und Ads-Engagement")
    plt.xlabel("Ads-Engagement (0 = kein, 3 = hoch)")
    plt.ylabel("Produktkategorie")
    plt.tight_layout()
    plt.show()

def categorize_age(df):
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ["<=20", "21–30", "31–40", "41–50", "51–60", "60+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    return df

def heatmap_ads_by_age_and_category(df: pd.DataFrame):
    df = get_ads_users(df)
    df = categorize_age(df)

    # Nur Produkte mit Return_Rate > 0 (nach deiner Vorgabe)
    filtered = df[df["Return_Rate"] > 0]

    # Pivot: Durchschnittliches Ads-Engagement nach Alter & Kategorie
    pivot = filtered.pivot_table(
        index="Purchase_Category",
        columns="Age_Group",
        values="Engagement_with_Ads",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("🎯 Ads-Engagement nach Alter & Produkt (nur Rückgaben)")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Produktkategorie")
    plt.tight_layout()
    plt.show()


def encode_purchase_intent(df):
    intent_map = {
        "Impulsive": 0,
        "Wants-based": 1,
        "Planned": 2,
        "Need-based": 3
    }
    df["Purchase_Intent_Encoded"] = df["Purchase_Intent"].replace(intent_map)
    return df


def heatmap_purchase_intent_by_age_and_category(df: pd.DataFrame):
    """Heatmap: Ø Kaufabsicht nach Altersgruppe & Produkt (nur bei Rückgabe > 0)."""
    df = get_ads_users(df)
    df = categorize_age(df)
    df = encode_purchase_intent(df)
    
    # Nur Produkte mit Rückgabequote > 0
    filtered = df[df["Return_Rate"] > 0]

    # Pivot: Mittelwert der kodierten Kaufabsicht
    pivot = filtered.pivot_table(
        index="Age_Group",
        columns="Purchase_Category",
        values="Purchase_Intent_Encoded",
        aggfunc="mean"
    )
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Oranges", linewidths=0.5)

    plt.title("🛍️ Kaufabsicht nach Altersgruppe & Produkt (nur bei Rückgabe)")
    plt.xlabel("Produktkategorie")
    plt.ylabel("Altersgruppe")
    plt.tight_layout()
    plt.show()

def plot_return_share_by_age_group(df: pd.DataFrame):
    df = get_ads_users(df)
    df = categorize_age(df)

    # Anzahl Rückgaben und Nutzer:innen pro Altersgruppe berechnen
    grouped = df.groupby("Age_Group").agg(
        total_users=("Return_Rate", "count"),
        total_returns=("Return_Rate", "sum")
    )

    # Rückgabe-Anteil in Prozent
    grouped["return_share_percent"] = (grouped["total_returns"] / grouped["total_users"]) * 100

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.index, y=grouped["return_share_percent"], color="tomato")
    plt.title("📦 Rückgabeanteil nach Altersgruppe (nur Ads-Nutzer:innen)")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Rückgabeanteil (%)")
    plt.tight_layout()
    plt.show()

def plot_ads_users_count_by_age_group(df: pd.DataFrame):
    df = get_ads_users(df)
    df = categorize_age(df)

    # Zähle Nutzer:innen pro Altersgruppe
    counts = df["Age_Group"].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, color="#1f77b4")
    plt.title("👥 Anzahl der Ads-Nutzer:innen pro Altersgruppe")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Anzahl Nutzer:innen")
    plt.tight_layout()
    plt.show()


def plot_purchase_frequency_by_age(df: pd.DataFrame):
    df = get_ads_users(df)
    df = categorize_age(df)

    grouped = df.groupby("Age_Group")["Frequency_of_Purchase"].mean()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.index, y=grouped.values, color="#ff7f0e")
    plt.title("🛒 Ø Kauffrequenz nach Altersgruppe (nur Ads-Nutzer:innen)")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Ø Kauffrequenz")
    plt.tight_layout()
    plt.show()

def plot_purchase_amount_by_age(df: pd.DataFrame):
    df = get_ads_users(df)
    df = categorize_age(df)

    grouped = df.groupby("Age_Group")["Purchase_Amount"].mean()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.index, y=grouped.values, color="#1f77b4")
    plt.title("💰 Ø Kaufbetrag nach Altersgruppe (nur Ads-Nutzer:innen)")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Ø Kaufbetrag")
    plt.tight_layout()
    plt.show()


def plot_total_revenue_by_age_group(df: pd.DataFrame):
    df = get_ads_users(df)
    df = categorize_age(df)

    # Gesamtumsatz je Altersgruppe berechnen
    grouped = df.groupby("Age_Group")["Purchase_Amount"].sum()

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.index, y=grouped.values, color="#2b8cbe")
    plt.title("💰 Gesamtumsatz nach Altersgruppe (nur Ads-Nutzer:innen)")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Gesamter Kaufbetrag (€)")
    plt.tight_layout()
    plt.show()


def heatmap_return_by_category_and_ads_per_age_group(df: pd.DataFrame):
    df = get_ads_users(df)
    df = categorize_age(df)

    age_groups = df["Age_Group"].unique().sort_values()

    for group in age_groups:
        subset = df[df["Age_Group"] == group]

        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="Purchase_Category",
            columns="Engagement_with_Ads",
            values="Return_Rate",
            aggfunc="mean"
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title(f"📦 Rückgabequote nach Produkt & Ads-Engagement – Altersgruppe {group}")
        plt.xlabel("Ads-Engagement (0 = kein, 3 = hoch)")
        plt.ylabel("Produktkategorie")
        plt.tight_layout()
        plt.show()
    
def heatmap_return_rate_by_location_and_category(df: pd.DataFrame):
    df = get_ads_users(df)

    pivot = df.pivot_table(
        index="Purchase_Category",
        columns="Purchase_Channel",
        values="Return_Rate",
        aggfunc="mean"
    )
    print("\n📊 Rückgabequote pro Standort (gesamt, in %):")
    location_returns = df.groupby("Purchase_Channel").agg(
        Rückgaben=("Return_Rate", "sum"),
        Käufe=("Return_Rate", "count")
    )
    location_returns["Rückgabequote (%)"] = (location_returns["Rückgaben"] / location_returns["Käufe"]) * 100
    print(location_returns[["Rückgabequote (%)"]].round(2))
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5)
    plt.title("📦 Ø Rückgabequote nach Produktkategorie & Standort (nur Ads-Nutzer:innen)")
    plt.xlabel("Standort")
    plt.ylabel("Produktkategorie")
    plt.tight_layout()
    plt.show()


