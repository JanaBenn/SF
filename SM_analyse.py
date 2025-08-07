import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import statsmodels.api as sm 


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
    
def scatter_vs_amount(df: pd.DataFrame, columns: list):
    for col in columns:
        plt.figure(figsize=(7, 5))
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.scatterplot(data=df, x=col, y="Purchase_Amount", alpha=0.6, color="#2ca02c")
        else:
            sns.boxplot(data=df, x=col, y="Purchase_Amount", palette="Set2")

        plt.title(f"Purchase Amount vs. {col}")
        plt.xlabel(col)
        plt.ylabel("Purchase Amount (USD)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        
        
def categorize_age(df):
    if "Age_Group" not in df.columns:
        bins = [0, 20, 30, 40, 50, 60]
        labels = ["<=20", "21–30", "31–40", "41–50", "51–60"]
        df.loc[:, "Age_Group"] = pd.cut(df["Age"], bins=bins, labels=pd.Categorical(labels, ordered=True), right=False)
    return df

       
def correlation_matrix_sm_return(df: pd.DataFrame, method: str = "spearman", show_heatmap: bool = True):
    """
    Berechnet Korrelationen zwischen Social Media Influence und Rückgabequote
    für jede Kombination aus Altersgruppe und Produktkategorie.
    
    :param df: Ursprünglicher DataFrame mit Social_Media_Influence und Return_Rate
    :param method: 'pearson' oder 'spearman'
    :param show_heatmap: Ob Heatmap angezeigt werden soll
    :return: Korrelationsmatrix als DataFrame
    """

    df = categorize_age(df)

    age_groups = df["Age_Group"].dropna().unique()
    categories = df["Purchase_Category"].dropna().unique()  

    results = []

    for age in age_groups:
        for category in categories:
            subset = df[
                (df["Age_Group"] == age) &
                (df["Purchase_Category"] == category)
            ]

            if len(subset) >= 5:  # Nur auswerten, wenn genug Datenpunkte
                corr_val = subset[["Social_Media_Influence", "Return_Rate"]].corr(method=method).iloc[0, 1]
            else:
                corr_val = np.nan  # Zu wenig Daten

            results.append({
                "Age_Group": age,
                "Purchase_Category": category,
                f"{method.capitalize()}_Corr": corr_val
            })

    corr_df = pd.DataFrame(results)

    # Optionale Heatmap
    if show_heatmap:
        pivot = corr_df.pivot(index="Purchase_Category", columns="Age_Group", values=f"{method.capitalize()}_Corr")
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
        plt.title(f"📊 {method.capitalize()}-Korrelation: SM ↔ Rückgabe")
        plt.xlabel("Altersgruppe")
        plt.ylabel("Produktkategorie")
        plt.tight_layout()
        plt.show()

    return corr_df



# REGRESSIONEN




def regression_agegroup_21_30_sm_logit(df: pd.DataFrame):
    """
    Führt eine logistische Regression durch:
    Return_Rate (0/1) ~ Social_Media_Influence
    für Altersgruppe 21–30
    """

    # Altersgruppen kategorisieren
    df = categorize_age(df)

    # Gefilterte Daten (nur 21–30)
    filtered = df[
        (df["Age_Group"] == "21–30") &
        (df["Return_Rate"].isin([0, 1])) &
        (df["Social_Media_Influence"].notna())
    ]
    
    print(f"🎯 Gefilterte Zeilen: {len(filtered)}")
    print(filtered.head())


    if len(filtered) < 10:
        print("⚠️ Zu wenige Datenpunkte für logistische Regression.")
        return None

    X = filtered[["Social_Media_Influence"]]
    X = sm.add_constant(X)
    y = filtered["Return_Rate"]

    try:
        model = sm.Logit(y, X).fit(disp=0)  # Kein zu langer Output
        print("📘 Logistische Regression: Return_Rate ~ Social_Media_Influence (Alter 21–30)")
        print(model.summary())
        return model
    except Exception as e:
        print("❌ Fehler bei der logistischen Regression:", str(e))
        return None
    
    
    
    
def regression_all_ages_logit(df: pd.DataFrame):
    """
    Führt eine logistische Regression durch:
    Return_Rate (0/1) ~ Social_Media_Influence
    für alle Altersgruppen.
    """

    # Nur gültige 0/1-Return-Raten und nicht-leere Social_Media_Influence
    filtered = df[
        df["Return_Rate"].isin([0, 1]) & 
        df["Social_Media_Influence"]>0
    ]

    print("Koeffizient -0.068: Mehr Social-Media-Influence ist leicht mit weniger Rückgaben assoziiert – aber nicht signifikant.")

    if len(filtered) < 20:
        print("⚠️ Zu wenige Datenpunkte für logistische Regression.")
        return None

    X = filtered[["Social_Media_Influence"]]
    X = sm.add_constant(X)
    y = filtered["Return_Rate"]

    try:
        model = sm.Logit(y, X).fit()
        print("📘 Logistische Regression: Return_Rate ~ Social_Media_Influence (ALLE Altersgruppen)")
        print(model.summary())
        return model
    except Exception as e:
        print("❌ Fehler bei der logistischen Regression:", str(e))
        return None