from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from typing import Optional, Iterable, Tuple, List
from statsmodels.tools.sm_exceptions import PerfectSeparationError


# -------------------------------------------------------------------
# Helpers – identisch/kompatibel zum ReturnLookupService
# -------------------------------------------------------------------

def _categorize_age(df: pd.DataFrame) -> pd.DataFrame:
    """Age -> Age_Group (<=20, 21–30, 31–40, 41–50, 51–60)."""
    bins = [0, 20, 30, 40, 50, 60]
    labels = ["<=20", "21–30", "31–40", "41–50", "51–60"]
    out = df.copy()
    out["Age_Group"] = pd.cut(
        out["Age"],
        bins=bins,
        labels=pd.Categorical(labels, categories=labels, ordered=True),
        right=False,
    )
    return out


def _categorize_income(df: pd.DataFrame) -> pd.DataFrame:
    """Income_Level -> Income_Label (robust für numerische Codes und bereits gelabelte Strings)."""
    out = df.copy()

    # Falls numerisch: 1/2/3 mappen
    if pd.api.types.is_numeric_dtype(out["Income_Level"]):
        income_map = {1: "Low", 2: "Middle", 3: "High"}
        out["Income_Label"] = out["Income_Level"].map(income_map)
    else:
        # Falls schon Strings: trimmen & vereinheitlichen
        canon = (
            out["Income_Level"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()  # z.B. "middle" -> "Middle"
        )
        # nur erlaubte Labels durchlassen
        canon = canon.where(canon.isin(["Low", "Middle", "High"]), other=np.nan)
        out["Income_Label"] = canon

    return out


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten im DataFrame: {missing}")


def _auto_detect_return_col(df: pd.DataFrame) -> str:
    """Versucht die Return-Spalte automatisch zu finden."""
    candidates = ["Return_Rate", "Return", "Returned", "is_return", "ReturnFlag"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Keine Return-Spalte gefunden. Erwartet eine der {candidates}. "
        f"Vorhandene Spalten: {list(df.columns)}"
    )


def _normalize_discount_series(s: pd.Series) -> pd.Series:
    """
    Discount_Used -> bool (robust: 1/0, t/f, true/false, yes/no).
    Unbekanntes → False.
    """
    if s.dtype == bool:
        return s.astype(bool)

    mapping = {
        "1": True, "0": False,
        "true": True, "false": False,
        "t": True, "f": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .map(mapping)
         .fillna(False)
         .astype(bool)
    )


# -------------------------------------------------------------------
# Modeling-Service
# -------------------------------------------------------------------

class ReturnModeler:
    """
    Kapselt Logit(Ridge)-Modellläufe pro Segment und liefert ORs als Long-DF.
    Erwartet Rohdaten; die nötigen Features werden intern wie im
    ReturnLookupService vorbereitet.
    """


    def __init__(self, df: pd.DataFrame, return_col: Optional[str] = None):
        """
        df: Rohdaten. Erwartet u.a. Age, Gender, Income_Level, Discount_Used, Purchase_Category, <Return-Spalte>
        return_col: Name der Return-Spalte (0/1). Wenn None -> Auto-Detection.
        """
        if return_col is None:
            return_col = _auto_detect_return_col(df)

        self.df = df.copy()
        self.return_col = return_col

        _ensure_cols(
            self.df,
            ["Age", "Gender", "Income_Level", "Discount_Used", "Purchase_Category", self.return_col],
        )

        # Kategorisieren
        self.df = _categorize_age(self.df)
        self.df = _categorize_income(self.df)

        # >>> WICHTIG: Nur Male/Female behalten
        self.df = self.df[self.df["Gender"].isin(["Male", "Female"])].copy()

        # Typen aufräumen
        self.df["Gender"] = self.df["Gender"].astype(str)
        self.df["Purchase_Category"] = self.df["Purchase_Category"].astype(str)

        # Discount_Used -> bool
        if self.df["Discount_Used"].dtype != bool:
            self.df["Discount_Used"] = (
                self.df["Discount_Used"]
                .astype(str).str.lower()
                .map({"1": True, "0": False, "true": True, "false": False})
                .fillna(False)
                .astype(bool)
            )

        # Return-Spalte binär (0/1) – zählt jede positive Zahl (inkl. 2) als Rückgabe
        self.df[self.return_col] = pd.to_numeric(self.df[self.return_col], errors="coerce").fillna(0.0)
        self.df[self.return_col] = (self.df[self.return_col] > 0).astype(float)

    # ---------------- Debug & Vorbereitung ----------------

    def debug_snapshot(self) -> pd.DataFrame:
        """
        Schneller Überblick über die wichtigen Dimensionen nach dem Preprocessing.
        """
        rows = [
            ("rows", len(self.df)),
            ("unique Gender", self.df["Gender"].nunique()),
            ("unique Income_Label", self.df["Income_Label"].nunique()),
            ("unique Discount_Used", self.df["Discount_Used"].nunique()),
            ("unique Age_Group", self.df["Age_Group"].nunique()),
            ("unique Purchase_Category", self.df["Purchase_Category"].nunique()),
            ("return_col", self.return_col),
            ("return_mean", float(self.df[self.return_col].mean()) if len(self.df) else np.nan),
        ]
        # zusätzlich: Häufigkeiten
        g_counts = self.df["Gender"].value_counts(dropna=False).to_dict()
        i_counts = self.df["Income_Label"].value_counts(dropna=False).to_dict()
        d_counts = self.df["Discount_Used"].value_counts(dropna=False).to_dict()
        rows.extend([
            ("Gender counts", g_counts),
            ("Income_Label counts", i_counts),
            ("Discount_Used counts", d_counts),
        ])
        return pd.DataFrame(rows, columns=["metric", "value"])

    def _build_group_df(self, gender: str, income_label: str, discount_used: bool) -> pd.DataFrame:
        """
        Schneidet die Daten auf ein (Gender, Income_Label, Discount_Used)-Segment zu.
        """
        tmp = self.df[
            (self.df["Gender"] == gender)
            & (self.df["Income_Label"] == income_label)
            & (self.df["Discount_Used"] == discount_used)
        ].dropna(subset=[self.return_col, "Purchase_Category", "Age_Group"]).copy()
        return tmp

    def debug_class_balance(self, combos: Iterable[Tuple[str, str, bool]]) -> pd.DataFrame:
        """
        Zeigt je Segment n, n0, n1.
        """
        rows: List[dict] = []
        for g, inc, disc in combos:
            sub = self._build_group_df(g, inc, disc)
            if sub.empty:
                rows.append({"Gender": g, "Income_Label": inc, "Discount_Used": disc, "n": 0, "n0": 0, "n1": 0})
                continue
            y = sub[self.return_col].astype(int)
            rows.append({
                "Gender": g, "Income_Label": inc, "Discount_Used": disc,
                "n": len(y), "n0": int((y == 0).sum()), "n1": int((y == 1).sum())
            })
        return pd.DataFrame(rows)

    # ---------------- Logit (Ridge) ----------------

    def _fit_logit_l2(
        self,
        temp_df: pd.DataFrame,
        alpha: float = 0.8,
        allow_single_class: bool = True,
    ):
        """
        Fit mit L2-Regularisierung. Bei nur einer Klasse optionaler Minimal-Fallback,
        damit das Modell stabil schätzt.
        """
        formula = f"{self.return_col} ~ C(Purchase_Category) + C(Age_Group)"
        y, X = patsy.dmatrices(formula, data=temp_df, return_type="dataframe")
        y = y.iloc[:, 0]  # Series

        if y.nunique() < 2:
            if not allow_single_class or X.shape[0] == 0:
                return None, None, None
            # Zwei Pseudo-Beobachtungen: eine mit y=0, eine mit y=1
            X0 = X.iloc[[0]].copy()
            X1 = X.iloc[[0]].copy()
            y_append = pd.Series([0.0, 1.0], index=[-1, -2])
            X = pd.concat([X, X0, X1], axis=0, ignore_index=True)
            y = pd.concat([y, y_append], axis=0, ignore_index=True)

        model = sm.Logit(y, X).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
        coef = pd.Series(model.params, index=X.columns)
        odds = np.exp(coef)
        return model, coef, odds

    def collect_odds_ratios(
        self,
        combos: Optional[Iterable[Tuple[str, str, bool]]] = None,
        alpha: float = 0.8,
        allow_single_class: bool = True,
    ) -> pd.DataFrame:
        """
        Läuft die Logit-Modelle über die Segmente und sammelt ORs (Long-DF).
        Spalten: {group, factor, OR}
        """
        if combos is None:
            combos = (
                ("Female", "High", True),   ("Female", "High", False),
                ("Female", "Middle", True), ("Female", "Middle", False),
                ("Male", "High", True),     ("Male", "High", False),
                ("Male", "Middle", True),   ("Male", "Middle", False),
            )

        rows: List[dict] = []

        for g, inc, disc in combos:
            sub = self._build_group_df(g, inc, disc)
            if sub.empty:
                print(f"[SKIP] {g} | {inc} | Disc={disc}: keine Daten nach Filter.")
                continue

            try:
                model, coef, odds = self._fit_logit_l2(
                    sub, alpha=alpha, allow_single_class=allow_single_class
                )
                if model is None:
                    print(f"[SKIP] {g} | {inc} | Disc={disc}: nur eine Klasse und allow_single_class=False.")
                    continue
            except (PerfectSeparationError, np.linalg.LinAlgError):
                print(f"[SKIP] {g} | {inc} | Disc={disc}: numerisches Problem.")
                continue
            except Exception as e:
                print(f"[SKIP] {g} | {inc} | Disc={disc}: {e}")
                continue

            group = f"{g} | {inc} | Disc={'True' if disc else 'False'}"

            for name, orv in odds.items():
                lname = name.lower()
                if "const" in lname or "intercept" in lname:
                    continue

                factor = None
                if name.startswith("C(Purchase_Category)"):
                    try:
                        factor = name.split("T.", 1)[1].rstrip("]")
                    except Exception:
                        factor = None
                elif name.startswith("C(Age_Group)"):
                    try:
                        lvl = name.split("T.", 1)[1].rstrip("]")
                        factor = f"Age {lvl}"
                    except Exception:
                        factor = None

                if factor is not None:
                    rows.append({"group": group, "factor": factor, "OR": float(orv)})

        return pd.DataFrame(rows) 