import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_ads_users(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt Nutzer mit tatsächlicher Ads-Interaktion zurück (Wert > 0)."""
    return df[df["Engagement_with_Ads"] > 0]

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
    plt.title("Verteilung der Werbe-Engagement-Level")
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
        "Brand_Loyalty"
    ]

    for col in target_columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x="Engagement_with_Ads", y=col, estimator='mean', errorbar=None)
        plt.title(f"{col} vs Engagement_with_Ads")
        plt.ylabel("Durchschnitt")
        plt.xlabel("Engagement mit Ads (0–3)")
        plt.tight_layout()
        plt.show()