import pandas as pd
import matplotlib.pyplot as plt

def get_sm_users(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt Nutzer mit Social-Media-Interaktion (ohne NaN) zurück."""
    return df[df["Social_Media_Influence"].notna()]


def plot_sm_user_insights(df: pd.DataFrame):
    sm_users = get_sm_users(df)

    # Spalten, die analysiert werden sollen
    columns_to_plot = [
        "Age", "Location", "Purchase_Category", "Purchase_Amount",
        "Frequency_of_Purchase", "Return_Rate", "Engagement_with_Ads", "Device_Used_for_Shopping",
        "Customer_Loyalty_Program_Member", "Customer_Satisfaction",
        "Product_Rating", "Brand_Loyalty"
    ]

    for col in columns_to_plot:
        if col not in sm_users.columns:
            print(f"⚠️ Spalte '{col}' nicht gefunden – wird übersprungen.")
            continue

        plt.figure(figsize=(8, 5))
        sm_users[col].value_counts(dropna=False).plot(kind='bar', color='#6baed6')
        plt.title(f"Social-Media-Nutzer: {col}")
        plt.xlabel(col)
        plt.ylabel("Anzahl")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()