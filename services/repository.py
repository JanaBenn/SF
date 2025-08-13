from __future__ import annotations
import pandas as pd

class DataRepository:
    """
    Verantwortlich fürs Laden & Grundreinigung der Rohdaten.
    (verlagert, was bisher in data_loader.py war)
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        print(f"📂 Lade Datei von: {self.path}")
        df = pd.read_csv(self.path)
        return df

    @staticmethod
    def categorize_age(df: pd.DataFrame) -> pd.DataFrame:
        bins = [0, 20, 30, 40, 50, 60]
        labels = ["<=20", "21–30", "31–40", "41–50", "51–60"]
        out = df.copy()
        out["Age_Group"] = pd.cut(
            out["Age"],
            bins=bins,
            labels=pd.Categorical(labels, categories=labels, ordered=True),
            right=False
        )
        return out

    @staticmethod
    def categorize_income(df: pd.DataFrame) -> pd.DataFrame:
        income_map = {1: "Low", 2: "Middle", 3: "High"}
        out = df.copy()
        out["Income_Label"] = out["Income_Level"].map(income_map)
        return out

    def load_and_prepare(self) -> pd.DataFrame:
        df = self.load()
        df = self.categorize_age(df)
        df = self.categorize_income(df)
        # ggf. weitere Bereinigungen hier zentralisieren
        return df


