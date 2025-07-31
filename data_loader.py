import pandas as pd
import os

def load_and_clean_data():
    base_dir = os.path.dirname(__file__)
    relative_path = os.path.join("customer_behavour", "Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    csv_path = os.path.join(base_dir, relative_path)
    
    print("📂 Lade Datei von:", csv_path)
    df = pd.read_csv(csv_path)
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print("\n🔎 Anzahl fehlender Werte pro Spalte:\n")
    print(df.isna().sum())

    missing_combinations = df[["Social_Media_Influence", "Engagement_with_Ads"]].isna()
    print(missing_combinations.value_counts())
    
    print("🔹 Social_Media_Influence:")
    print(df["Social_Media_Influence"]) 

    print("🔹 Engagement_with_Ads:")
    print(df["Engagement_with_Ads"])

    
    

    

