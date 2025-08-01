import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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