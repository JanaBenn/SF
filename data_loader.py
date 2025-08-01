import pandas as pd
import os

def load_data():
    base_dir = os.path.dirname(__file__)
    relative_path = os.path.join("customer_behavour", "Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    csv_path = os.path.join(base_dir, relative_path)
    
    print("📂 Lade Datei von:", csv_path)
    df = pd.read_csv(csv_path)
    return df


# 👉 Cleaning-Funktion
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Purchase_Amount: $ entfernen & float
    df["Purchase_Amount"] = df["Purchase_Amount"].replace(r"[\$,]", "", regex=True).astype(float)

    # 2. Kategorie-Mapping für Social Media & Ads
    influence_map = {
        "High": 3,
        "Medium": 2,
        "Low": 1,
        pd.NA: 0,
        None: 0,
        float("nan"): 0
    }

    df["Social_Media_Influence"] = df["Social_Media_Influence"].map(influence_map).fillna(0).astype(int)
    df["Engagement_with_Ads"] = df["Engagement_with_Ads"].map(influence_map).fillna(0).astype(int)

    # 3. Time_of_Purchase → datetime
    df["Time_of_Purchase"] = pd.to_datetime(df["Time_of_Purchase"], errors="coerce")

    # 4. Spalte umbenennen
    df.rename(columns={
        "Time_Spent_on_Product_Research(hours)": "Time_Spent_on_Research"
    }, inplace=True)

    return df


# 👉 Einheitliche Funktion für externen Import
def load_and_clean_data() -> pd.DataFrame:
    df = load_data()
    return clean_data(df)


# 👉 Nur zu Testzwecken, wenn direkt ausgeführt
if __name__ == "__main__":
    df = load_and_clean_data()

    print("\n🔎 Anzahl fehlender Werte pro Spalte:\n")
    print(df.isna().sum())

    print("\n📊 Spalten und Datentypen:\n")
    print(df.dtypes)