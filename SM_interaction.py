import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_sm_users(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt Nutzer:innen mit Social-Media-Interaktion zurück, aber ohne Ads-Engagement."""
    return df[
        (df["Social_Media_Influence"].notna()) &
        (df["Social_Media_Influence"] > 0) &
        (
            df["Engagement_with_Ads"].isna() |
            (df["Engagement_with_Ads"] == 0)
        )
    ]


def plot_sm_user_insights(df: pd.DataFrame):
    sm_users = get_sm_users(df)

    # Spalten für Analyse
    columns_to_plot = [
        "Age", "Gender",
        "Income_Level","Purchase_Category", "Purchase_Amount",
        "Frequency_of_Purchase", "Return_Rate", "Device_Used_for_Shopping",
        "Customer_Satisfaction",
        "Product_Rating", "Brand_Loyalty"
    ]

    for col in columns_to_plot:
        if col not in sm_users.columns:
            print(f"⚠️ Spalte '{col}' nicht gefunden – wird übersprungen.")
            continue

        plt.figure(figsize=(8, 5))

        # Wähle Plot-Typ abhängig vom Datentyp
        if pd.api.types.is_numeric_dtype(sm_users[col]):
            sm_users[col].plot(kind='hist', bins=10, color='#6baed6', edgecolor='black')
            plt.ylabel("Anzahl")
        else:
            sm_users[col].value_counts(dropna=False).plot(kind='bar', color='#6baed6')
            plt.ylabel("Anzahl")

        plt.title(f"Social-Media-Nutzer: {col}")
        plt.xlabel(col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()



def describe_sm_users(df: pd.DataFrame):
    sm_users = get_sm_users(df)
    print("\n📊 Anzahl SM-Nutzer (Influence > 0, ohne Ads):", len(sm_users))
    print("\n📈 Verteilung der Social-Media-Influence-Werte:")
    print(df["Social_Media_Influence"].value_counts().sort_index())
    print("\n🔍 Vorschau:")
    print(sm_users.head())

def plot_sm_engagement_distribution(df: pd.DataFrame):
    counts = df["Social_Media_Influence"].value_counts().sort_index()
    x_labels = [str(i) for i in counts.index]
    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, counts.values, color="#4daf4a")
    plt.title("Verteilung des Social-Media-Influence")
    plt.xlabel("Influence-Level")
    plt.ylabel("Anzahl Nutzer") 
    plt.tight_layout()
    plt.show()

def barplots_vs_sm(df: pd.DataFrame):
    sm_users = get_sm_users(df)
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
        sns.barplot(data=sm_users, x="Social_Media_Influence", y=col, estimator='mean', errorbar=None)
        plt.title(f"{col} vs Social Media Influence")
        plt.ylabel("Durchschnitt")
        plt.xlabel("Social Media Influence (1–3)")
        plt.tight_layout()
        plt.show()

def barplots_intent_vs_targets(df: pd.DataFrame):
    sm_users = get_sm_users(df)
    target_columns = ["Return_Rate", "Customer_Satisfaction"]
    for col in target_columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=sm_users, x="Purchase_Intent", y=col, estimator="mean", errorbar=None)
        plt.title(f"{col} vs Purchase Intent (nur SM-Nutzer)")
        plt.xlabel("Kaufabsicht (0–3)")
        plt.ylabel("Durchschnitt")
        plt.tight_layout()
        plt.show()

def barplots_discount_vs_targets(df: pd.DataFrame):
    sm_users = get_sm_users(df)
    target_columns = ["Customer_Satisfaction", "Return_Rate"]
    for col in target_columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=sm_users, x="Discount_Used", y=col, estimator="mean", errorbar=None)
        plt.title(f"{col} vs Discount Used (nur SM-Nutzer)")
        plt.xlabel("Discount genutzt (0 = nein, 1 = ja)")
        plt.ylabel("Durchschnitt")
        plt.tight_layout()
        plt.show()

def categorize_age(df):
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ["<=20", "21–30", "31–40", "41–50", "51–60", "60+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    return df

def encode_purchase_intent(df):
    intent_map = {
        "Impulsive": 0,
        "Wants-based": 1,
        "Planned": 2,
        "Need-based": 3
    }
    df["Purchase_Intent_Encoded"] = df["Purchase_Intent"].replace(intent_map)
    return df

def heatmap_return_by_category_and_sm_per_age_group(df: pd.DataFrame):
    df = get_sm_users(df)
    df = categorize_age(df)
    age_groups = df["Age_Group"].unique().sort_values()
    for group in age_groups:
        subset = df[df["Age_Group"] == group]
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index="Purchase_Category",
            columns="Social_Media_Influence",
            values="Return_Rate",
            aggfunc="mean"
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title(f"📦 Rückgabequote nach Produkt & SM-Influence – Altersgruppe {group}")
        plt.xlabel("SM-Influence (1–3)")
        plt.ylabel("Produktkategorie")
        plt.tight_layout()
        plt.show()

def heatmap_return_rate_by_location_and_category(df: pd.DataFrame):
    df = get_sm_users(df)
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
    plt.title("📦 Ø Rückgabequote nach Produktkategorie & Standort (nur SM-Nutzer:innen)")
    plt.xlabel("Standort")
    plt.ylabel("Produktkategorie")
    plt.tight_layout()
    plt.show()

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



def heatmap_return_by_category_and_SM(df: pd.DataFrame):
    """Heatmap: Rückgabequote je Produktkategorie und SM-Engagement."""
    sm_users = get_sm_users(df)

    # Gruppieren: mittlere Rückgabequote pro Kategorie und Engagement-Level
    pivot = sm_users.pivot_table(
        index="Purchase_Category",
        columns="Social_Media_Influence",
        values="Return_Rate",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", linewidths=0.5)
    plt.title("📦 Rückgabequote nach Produktkategorie und SM-Engagement")
    plt.xlabel("SM-Engagement (1 = niedrig, 3 = hoch)")
    plt.ylabel("Produktkategorie")
    plt.tight_layout()
    plt.show()


def heatmap_SM_by_age_and_category(df: pd.DataFrame):
    df = get_sm_users(df)
    df = categorize_age(df)

    # Nur Produkte mit Return_Rate > 0 (nach deiner Vorgabe)
    filtered = df[df["Return_Rate"].isin([0, 1])]

    # Pivot: Durchschnittliches Ads-Engagement nach Alter & Kategorie
    pivot = filtered.pivot_table(
        index="Purchase_Category",
        columns="Age_Group",
        values="Social_Media_Influence",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("🎯 SM-Engagement nach Alter & Produkt")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Produktkategorie")
    plt.tight_layout()
    plt.show()


def heatmap_Return_by_age_and_category(df: pd.DataFrame):
    df = get_sm_users(df)
    df = categorize_age(df)

    # Nur Produkte mit Return_Rate > 0 (nach deiner Vorgabe)
    filtered = df[df["Return_Rate"].isin([0, 1])]

    # Pivot: Durchschnittliches Ads-Engagement nach Alter & Kategorie
    pivot = filtered.pivot_table(
        index="Purchase_Category",
        columns="Age_Group",
        values="Return_Rate",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("🎯 Return nach Alter & Produkt")
    plt.xlabel("Altersgruppe")
    plt.ylabel("Produktkategorie")
    plt.tight_layout()
    plt.show()





def analyze_and_plot_return_sm_correlation(df: pd.DataFrame):
    """
    Kombiniert Rückgabe- und SM-Heatmaps: 
    Identifiziert Kombinationen mit (a) hoher Rückgabe & hohem SM-Engagement 
    und (b) keinem Rückgabeverhalten trotz hoher SM-Nutzung.
    Zeigt beide Muster in Balkendiagrammen.
    """

    # Daten vorbereiten
    sm_users = get_sm_users(df)
    sm_users = categorize_age(sm_users)

    # Pivot-Tabellen erzeugen
    return_pivot = sm_users.pivot_table(
        index="Purchase_Category",
        columns="Age_Group",
        values="Return_Rate",
        aggfunc="mean"
    )

    sm_pivot = sm_users.pivot_table(
        index="Purchase_Category",
        columns="Age_Group",
        values="Social_Media_Influence",
        aggfunc="mean"
    )

    # Pivot-DataFrames mergen (in Long-Form)
    merged = return_pivot.stack().to_frame("Return_Rate").join(
        sm_pivot.stack().to_frame("SM_Engagement")
    ).reset_index()

    merged.columns = ["Purchase_Category", "Age_Group", "Return_Rate", "SM_Engagement"]

    # 🟥 Fall 1: Hohe Rückgabe und hohes SM-Engagement
    high_corr = merged[
        (merged["Return_Rate"] == 1.0) &
        (merged["SM_Engagement"] >= 2.5)
    ]

    # 🟦 Fall 2: Keine Rückgabe, aber trotzdem hohes SM-Engagement
    low_corr = merged[
        (merged["Return_Rate"] == 0.0) &
        (merged["SM_Engagement"] >= 2.0)
    ]

    # Plot 1: Positive Korrelation
    plt.figure(figsize=(12, 6))
    sns.barplot(data=high_corr, x="Age_Group", y="SM_Engagement", hue="Purchase_Category")
    plt.title("📈 Hohe Rückgabequote & Hohes SM-Engagement")
    plt.ylabel("Ø SM-Influence")
    plt.xlabel("Altersgruppe")
    plt.legend(title="Produktkategorie", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot 2: Negative Korrelation
    plt.figure(figsize=(12, 6))
    sns.barplot(data=low_corr, x="Age_Group", y="SM_Engagement", hue="Purchase_Category")
    plt.title("📉 Keine Rückgabe trotz hohem SM-Engagement")
    plt.ylabel("Ø SM-Influence")
    plt.xlabel("Altersgruppe")
    plt.legend(title="Produktkategorie", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    try:
        from IPython.display import display
        print("🔍 Tabelle: Hohe Rückgaben & hohes SM:")
        display(high_corr)
        print("\n🔍 Tabelle: Keine Rückgaben trotz hohem SM:")
        display(low_corr)
    except ImportError:
        pass  # Kein IPython? Kein Problem.

    # 📤 Rückgabe der DataFrames zur weiteren Nutzung
    return high_corr, low_corr




def analyze_and_plot_return_sm_NEG_correlation(df: pd.DataFrame):
    """
    Kombiniert Rückgabe- und SM-Heatmaps: 
    Identifiziert Kombinationen mit (a) hoher Rückgabe & NIEDRIGEN SM-Engagement 
    und (b) keinem Rückgabeverhalten trotz NIEDRIGER SM-Nutzung.
    Zeigt beide Muster in Balkendiagrammen.
    """

    # Daten vorbereiten
    sm_users = get_sm_users(df)
    sm_users = categorize_age(sm_users)

    # Pivot-Tabellen erzeugen
    return_pivot = sm_users.pivot_table(
        index="Purchase_Category",
        columns="Age_Group",
        values="Return_Rate",
        aggfunc="mean"
    )

    sm_pivot = sm_users.pivot_table(
        index="Purchase_Category",
        columns="Age_Group",
        values="Social_Media_Influence",
        aggfunc="mean"
    )

    # Pivot-DataFrames mergen (in Long-Form)
    merged = return_pivot.stack().to_frame("Return_Rate").join(
        sm_pivot.stack().to_frame("SM_Engagement")
    ).reset_index()

    merged.columns = ["Purchase_Category", "Age_Group", "Return_Rate", "SM_Engagement"]

    # 🟥 Fall 1: Hohe Rückgabe und niedriges SM-Engagement
    high_corr = merged[
        (merged["Return_Rate"] == 1.0) &
        (merged["SM_Engagement"] < 2.5)
    ]

    # 🟦 Fall 2: Keine Rückgabe, und niedriges SM-Engagement
    low_corr = merged[
        (merged["Return_Rate"] == 0.0) &
        (merged["SM_Engagement"] < 2.0)
    ]

    # Plot 1: Positive Korrelation
    plt.figure(figsize=(12, 6))
    sns.barplot(data=high_corr, x="Age_Group", y="SM_Engagement", hue="Purchase_Category")
    plt.title("📈 Hohe Rückgabequote & NIEDRIGES SM-Engagement")
    plt.ylabel("Ø SM-Influence")
    plt.xlabel("Altersgruppe")
    plt.legend(title="Produktkategorie", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot 2: Negative Korrelation
    plt.figure(figsize=(12, 6))
    sns.barplot(data=low_corr, x="Age_Group", y="SM_Engagement", hue="Purchase_Category")
    plt.title("📉 Keine Rückgabe und NIEDRIGES SM-Engagement")
    plt.ylabel("Ø SM-Influence")
    plt.xlabel("Altersgruppe")
    plt.legend(title="Produktkategorie", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    try:
        from IPython.display import display
        print("🔍 Tabelle: Hohe Rückgaben & NIEDRIGES SM:")
        display(high_corr)
        print("\n🔍 Tabelle: Keine Rückgaben und NIEDRIGES SM:")
        display(low_corr)
    except ImportError:
        pass  # Kein IPython? Kein Problem.

    # 📤 Rückgabe der DataFrames zur weiteren Nutzung
    return high_corr, low_corr


def correlation_sm_return_overall(df: pd.DataFrame):
    # Altersgruppe vorbereiten
    df = categorize_age(df)

    # Nur 21–30-Jährige filtern
    young_adults = df[df["Age_Group"] == "21–30"]

    # Korrelationen berechnen
    pearson_corr = young_adults[["Social_Media_Influence", "Return_Rate"]].corr(method="pearson").iloc[0, 1]
    spearman_corr = young_adults[["Social_Media_Influence", "Return_Rate"]].corr(method="spearman").iloc[0, 1]

    print("📊 Korrelation SM ↔ Rückgabe (Altersgruppe 21–30):")
    print(f"• Pearson:  {pearson_corr:.2f}")
    print(f"• Spearman: {spearman_corr:.2f}")

    return pearson_corr, spearman_corr


def correlation_sm_return_furniture(df: pd.DataFrame):
    # Altersgruppe vorbereiten
    df = categorize_age(df)

    # Filter: 21–30 Jahre & Produktkategorie = Furniture
    filtered = df[
        (df["Age_Group"] == "21–30") &
        (df["Purchase_Category"] == "Furniture")
    ]

    # Korrelationen berechnen
    pearson_corr = filtered[["Social_Media_Influence", "Return_Rate"]].corr(method="pearson").iloc[0, 1]
    spearman_corr = filtered[["Social_Media_Influence", "Return_Rate"]].corr(method="spearman").iloc[0, 1]

    print("🪑 Korrelation SM ↔ Rückgabe (Furniture, 21–30):")
    print(f"• Pearson:  {pearson_corr:.2f}")
    print(f"• Spearman: {spearman_corr:.2f}")

    return pearson_corr, spearman_corr




