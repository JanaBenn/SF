from data_loader import load_and_clean_data 
from AD_interaction import barplots_vs_ads
from visualisation import plot_interaction_overview
from AD_interaction import plot_ads_engagement_distribution

def main():
    
    df = load_and_clean_data()
    barplots_vs_ads(df)
    plot_interaction_overview(df)
    plot_ads_engagement_distribution(df)


if __name__ == "__main__":
    main()
    
