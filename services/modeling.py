from __future__ import annotations
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from typing import Optional, Iterable
from statsmodels.tools.sm_exceptions import PerfectSeparationError

class ReturnModeler:
    """
    Kapselt Logit(Ridge)-Modellläufe pro Segment und liefert ORs als Long-DF.
    Erwartet, dass df Age_Group & Income_Label bereits besitzt.
    """

    def __init__(self, df_prepared: pd.DataFrame):
        self.df = df_prepared.copy()

    @staticmethod
    def _build_group_df(df, gender, income_label, discount_used) -> pd.DataFrame:
        tmp = df[
            (df["Gender"] == gender) &
            (df["Income_Label"] == income_label) &
            (df["Discount_Used"] == discount_used)
        ].dropna(subset=["Return_Rate","Purchase_Category","Age_Group"]).copy()
        if tmp.empty or tmp["Return_Rate"].nunique() < 2:
            return pd.DataFrame()
        tmp["Return_Rate"] = tmp["Return_Rate"].astype(float)
        return tmp

    @staticmethod
    def _fit_logit_l2(temp_df: pd.DataFrame, alpha: float = 0.8):
        y, X = patsy.dmatrices(
            "Return_Rate ~ C(Purchase_Category) + C(Age_Group)",
            data=temp_df,
            return_type="dataframe"
        )
        model = sm.Logit(y, X).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
        coef = pd.Series(model.params, index=X.columns)
        odds = np.exp(coef)
        return model, coef, odds

    def collect_odds_ratios(
        self,
        combos: Optional[Iterable[tuple[str,str,bool]]] = None,
        alpha: float = 0.8
    ) -> pd.DataFrame:
        if combos is None:
            combos = (
                ("Female","High",True),   ("Female","High",False),
                ("Female","Middle",True), ("Female","Middle",False),
                ("Male","High",True),     ("Male","High",False),
                ("Male","Middle",True),   ("Male","Middle",False),
            )
        rows = []
        for g, inc, disc in combos:
            sub = self._build_group_df(self.df, g, inc, disc)
            if sub.empty:
                continue
            try:
                _, _, odds = self._fit_logit_l2(sub, alpha=alpha)
            except (PerfectSeparationError, np.linalg.LinAlgError):
                continue
            group = f"{g} | {inc} | Disc={'True' if disc else 'False'}"
            for name, orv in odds.items():
                if name.lower() in ("const","intercept"):
                    continue
                factor = None
                if name.startswith("C(Purchase_Category)"):
                    try: factor = name.split("T.",1)[1].rstrip("]")
                    except: continue
                elif name.startswith("C(Age_Group)"):
                    try:
                        lvl = name.split("T.",1)[1].rstrip("]")
                        factor = f"Age {lvl}"
                    except: continue
                if factor is not None:
                    rows.append({"group": group, "factor": factor, "OR": float(orv)})
        return pd.DataFrame(rows)
