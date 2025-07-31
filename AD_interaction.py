import pandas as pd

def get_ads_users(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt Nutzer mit Ads-Interaktion (ohne NaN) zurück."""
    return df[df["Engagement_with_Ads"].notna()]

def describe_ads_users(df: pd.DataFrame):
    ads_users = get_ads_users(df)
    print("\n📊 Anzahl Ads-Nutzer:", len(ads_users))
    print("\n🔍 Vorschau:")
    print(ads_users.head())