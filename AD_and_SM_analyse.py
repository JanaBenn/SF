import pandas as pd
import statsmodels.api as sm

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


def categorize_age(df):
    if "Age_Group" not in df.columns:
        bins = [0, 20, 30, 40, 50, 60]
        labels = ["<=20", "21–30", "31–40", "41–50", "51–60"]
        df.loc[:, "Age_Group"] = pd.cut(
            df["Age"],
            bins=bins,
            labels=pd.Categorical(labels, ordered=True),
            right=False
        )
    return df


# --- REGRESSIONSMODELLE ---


def LR_return_vs_ads_and_SM(df: pd.DataFrame):
    """
    Logistische Regression: Rückgabequote ~ Engagement_with_Ads + Social_Media_Influence
    Nur sinnvoll bei binärer Zielvariable (0 = keine Rückgabe, 1 = Rückgabe).
    """
    ads_users = get_ads_users(df)

    filtered = ads_users[
        ads_users["Return_Rate"].isin([0, 1]) &
        ads_users["Engagement_with_Ads"].notna() &
        ads_users["Social_Media_Influence"].notna()
    ]

    if len(filtered) < 10:
        print("⚠️ Zu wenige Datenpunkte für logistische Regression.")
        return None

    X = filtered[["Engagement_with_Ads", "Social_Media_Influence"]]
    X = sm.add_constant(X)
    y = filtered["Return_Rate"]

    try:
        model = sm.Logit(y, X).fit(disp=0)
        print("\n📘 Logistische Regression: Return_Rate ~ Ads + SM Influence")
        print(model.summary())
        return model
    except Exception as e:
        print("❌ Fehler bei der logistischen Regression:", str(e))
        return None


def LR_ads_AND_SM_vs_return_by_age(df: pd.DataFrame):
    """
    Führt logistische Regressionen für jede Altersgruppe durch:
    Return_Rate ~ Engagement_with_Ads + Social_Media_Influence
    """
    df = categorize_age(df)
    age_groups = df["Age_Group"].dropna().unique()

    for age_group in age_groups:
        subset = df[
            (df["Age_Group"] == age_group) &
            (df["Return_Rate"].isin([0, 1])) &
            (df["Engagement_with_Ads"].notna()) &
            (df["Social_Media_Influence"].notna())
        ]

        print(f"\n📂 Altersgruppe: {age_group} – Beobachtungen: {len(subset)}")

        if len(subset) < 10:
            print("⚠️ Zu wenige Datenpunkte – Regression übersprungen.")
            continue

        X = subset[["Engagement_with_Ads", "Social_Media_Influence"]]
        X = sm.add_constant(X)
        y = subset["Return_Rate"]

        try:
            model = sm.Logit(y, X).fit(disp=0)
            print(model.summary())
        except Exception as e:
            print(f"❌ Fehler bei der Regression: {e}")
            
    print("Konsummenge der Ads und Social Media hat keinen nachweisbaren Einfluss auf Rückgaben – weder insgesamt noch in einzelnen Altersgruppen")
    



def run_agegroup_logit_regressions(df: pd.DataFrame) -> None:
    """
    Logistische Regression auf Rückgabequote nach Altersgruppen.
    Zielvariable: Return_Rate (binär: 0 = keine Rückgabe, 1 = Rückgabe)
    """
    df = categorize_age(df)

    age_groups = df["Age_Group"].dropna().unique()

    for age_group in age_groups:
        print(f"\n📂 Altersgruppe: {age_group}")

        # Daten für Altersgruppe filtern
        df_group = df[df["Age_Group"] == age_group].copy()

        if len(df_group) < 30:
            print(f"⚠️ Zu wenige Beobachtungen ({len(df_group)}). Übersprungen.")
            continue

        # Nur relevante Spalten
        cols_needed = [
            "Return_Rate",
            "Engagement_with_Ads",
            "Social_Media_Influence",
            "Purchase_Amount",
            "Income_Level"
        ]
        df_group = df_group.dropna(subset=cols_needed)

        # Interaktionsterm
        df_group["Interaction"] = (
            df_group["Engagement_with_Ads"] * df_group["Social_Media_Influence"]
        )

        # Feature Columns
        feature_cols = [
            "Engagement_with_Ads",
            "Social_Media_Influence",
            "Interaction",
            "Purchase_Amount"
        ] + [
            col for col in df_group.columns
            if col.startswith("Income_Level_") or col.startswith("Purchase_Category_")
        ]

        X = df_group[feature_cols]
        y = df_group["Return_Rate"]

        # Konstante hinzufügen
        X = sm.add_constant(X)

        try:
            model = sm.Logit(y, X).fit(disp=False)
            print(model.summary())
        except Exception as e:
            print(f"❌ Fehler bei Altersgruppe {age_group}: {e}")