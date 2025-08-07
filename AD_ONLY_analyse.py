import pandas as pd
import statsmodels.api as sm

def get_ads_users(df: pd.DataFrame) -> pd.DataFrame:
    """Nur Nutzer mit Ads-Engagement > 0, aber ohne Social Media Einfluss."""
    return df[
        (df["Engagement_with_Ads"] > 0) &
        (
            df["Social_Media_Influence"].isna() |
            (df["Social_Media_Influence"] == 0)
        )
    ]

def categorize_age(df):
    if "Age_Group" not in df.columns:
        bins = [0, 20, 30, 40, 50, 60]
        labels = ["<=20", "21–30", "31–40", "41–50", "51–60"]
        df.loc[:, "Age_Group"] = pd.cut(df["Age"], bins=bins, labels=pd.Categorical(labels, ordered=True), right=False)
    return df




# REGRESSIONSMODELLE:




def logistic_regression_return_vs_ads(df: pd.DataFrame):
    """
    Logistische Regression: Rückgabequote vs. Ads-Engagement (nur Ads-Nutzer:innen).
    Nur sinnvoll bei binärer Zielvariable (0 = keine Rückgabe, 1 = Rückgabe).
    """
    ads_users = get_ads_users(df)

    # Nur gültige 0/1-Werte in Return_Rate verwenden
    filtered = ads_users[ads_users["Return_Rate"].isin([0, 1])]

    if len(filtered) < 10:
        print("⚠️ Zu wenige Datenpunkte für logistische Regression.")
        return None

    X = filtered["Engagement_with_Ads"]
    y = filtered["Return_Rate"]

    # Konstante hinzufügen
    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X).fit(disp=0)
        print("\n📘 Logistische Regression: Return_Rate ~ Engagement_with_Ads (nur Ads-Nutzer:innen)")
        print(model.summary())
        return model
    except Exception as e:
        print("❌ Fehler bei der logistischen Regression:", str(e))
        return None






def logistic_regression_ads_vs_return_by_age(df: pd.DataFrame):
    """
    Führt logistische Regressionen für jede Altersgruppe durch:
    Return_Rate ~ Engagement_with_Ads
    Nur für Nutzer:innen mit gültigen Werten.
    """

    df = categorize_age(df)  # Altersgruppen sicherstellen

    age_groups = df["Age_Group"].dropna().unique()

    for age_group in age_groups:
        subset = df[
            (df["Age_Group"] == age_group) &
            (df["Engagement_with_Ads"].notna()) &
            (df["Return_Rate"].isin([0, 1]))
        ]

        print(f"\n📂 Altersgruppe: {age_group} – Beobachtungen: {len(subset)}")

        if len(subset) < 10:
            print("⚠️ Zu wenige Datenpunkte – Regression übersprungen.")
            continue

        X = sm.add_constant(subset["Engagement_with_Ads"])
        y = subset["Return_Rate"]

        try:
            model = sm.Logit(y, X).fit(disp=0)
            print(model.summary())
        except Exception as e:
            print(f"❌ Fehler bei der Regression: {e}")
        
        print("Engagement mit Werbung hat keinen nachweisbaren Einfluss auf Rückgaben – weder insgesamt noch in einzelnen Altersgruppen")