import pandas as pd
import matplotlib.pyplot as plt

def get_sm_users(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt Nutzer mit Social-Media-Interaktion (ohne NaN) zurück."""
    return df[df["Social_Media_Influence"]]


# def plot_sm_user_insights(df: pd.DataFrame):
    # sm_users = get_sm_users(df)

            # ALLGEMEIN ÜBERSICHT: Spalten, die analysiert werden sollen
      #columns_to_plot = [
         #  "Age", "Location", "Purchase_Category", "Purchase_Amount",
        #  "Frequency_of_Purchase", "Return_Rate", "Engagement_with_Ads", "Device_Used_for_Shopping",
        #   "Customer_Loyalty_Program_Member", "Customer_Satisfaction",
         #  "Product_Rating", "Brand_Loyalty"
    # ]

     # for col in columns_to_plot:
        #  if col not in sm_users.columns:
             # print(f"⚠️ Spalte '{col}' nicht gefunden – wird übersprungen.")
             # continue

         # plt.figure(figsize=(8, 5))
         # sm_users[col].value_counts(dropna=False).plot(kind='bar', color='#6baed6')
        #  plt.title(f"Social-Media-Nutzer: {col}")
        #  plt.xlabel(col)
         # plt.ylabel("Anzahl")
        #  plt.xticks(rotation=45, ha="right")
        #  plt.tight_layout()
         # plt.show()
         
def plot_top_bottom_filtered(series: pd.Series, colname: str, top_n=5, bottom_n=3):
    value_counts = series.value_counts().sort_values(ascending=False)

    # Top & Bottom nach Häufigkeit
    top = value_counts.head(top_n)
    bottom = value_counts.tail(bottom_n)

    # Kombinieren & sortiert anzeigen
    combined = pd.concat([top, bottom])

    colors = ["#2ca02c"] * len(top) + ["#d62728"] * len(bottom)

    plt.figure(figsize=(8, 5))
    combined.plot(kind="bar", color=colors)
    plt.title(f"Top {top_n} & Bottom {bottom_n} – {colname}")
    plt.ylabel("Anzahl Nutzer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_sm_top_bottom_filtered(df: pd.DataFrame):
    sm_users = get_sm_users(df)

    # Kategorische Spalte: Purchase_Category (Top 5 / Flop 3)
    plot_top_bottom_filtered(sm_users["Purchase_Category"], "Purchase_Category", top_n=5, bottom_n=3)

    # Diskrete Werte: Purchase_Amount, Frequency_of_Purchase, Age, Location
    plot_top_bottom_filtered(sm_users["Frequency_of_Purchase"], "Frequency_of_Purchase", top_n=3, bottom_n=3)
    plot_top_bottom_filtered(sm_users["Location"], "Location", top_n=3, bottom_n=3)
    plot_top_bottom_filtered(sm_users["Age"], "Age", top_n=3, bottom_n=3)


def print_top_values(series: pd.Series, colname: str, top_n=10):
    sorted_series = series.sort_values(ascending=False).head(top_n)
    print(f"\n📋 Top {top_n} Werte in '{colname}':\n")
    for i, (index, value) in enumerate(sorted_series.items(), start=1):
        print(f"{i}. Index {index}: {value:.2f}")

def print_bottom_values(series: pd.Series, colname: str, bottom_n=10):
    sorted_series = series.sort_values(ascending=True).head(bottom_n)
    print(f"\n📋 Bottom {bottom_n} Werte in '{colname}':\n")
    for i, (index, value) in enumerate(sorted_series.items(), start=1):
        print(f"{i}. Index {index}: {value:.2f}")

def plot_amount_distribution(df: pd.DataFrame, colname="Purchase_Amount"):
    bins = [0, 100, 200, 300, df[colname].max()]
    labels = ["50–100", "100–200", "200–300", "300–500"]

    # Werte den Kategorien (Bins) zuweisen
    df["Amount_Range"] = pd.cut(df[colname], bins=bins, labels=labels, right=True, include_lowest=True)

    # Häufigkeit pro Kategorie zählen
    range_counts = df["Amount_Range"].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(8, 5))
    range_counts.plot(kind="bar", color="#6baed6")
    plt.title("Verteilung der Transaktionen nach Betrag")
    plt.xlabel("Betragsbereich (USD)")
    plt.ylabel("Anzahl Transaktionen")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Optional: Textausgabe
    print("\n📊 Anzahl Transaktionen pro Bereich:")
    print(range_counts)