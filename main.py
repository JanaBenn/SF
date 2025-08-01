from data_loader import load_and_clean_data
from SM_interaction import plot_sm_user_insights
from AD_interaction import describe_ads_users
from visualization import plot_interaction_overview

def main():
    df = load_and_clean_data()
    describe_ads_users(df)
    plot_sm_user_insights(df)
    plot_interaction_overview(df)

if __name__ == "__main__":
    main()
    
