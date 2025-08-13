from services.repository import DataRepository
from services.returns import ReturnLookupService

def test_build_lookup_smoke():
    repo = DataRepository("customer_behavour/Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    dfp = repo.load_and_prepare()
    rl = ReturnLookupService(dfp)
    lookup = rl.build_lookup()
    assert {"Return_%","total_purchases","total_returns"}.issubset(set(lookup.columns))