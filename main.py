from data_loader import load_and_clean_data
from visualisation import plot_interaction_overview, plot_return_logit_summary
from Return import collect_odds_ratios


def main():
    
    df = load_and_clean_data() 
    plot_interaction_overview(df)
    or_df = collect_odds_ratios(df, alpha=0.8)
    plot_return_logit_summary(or_df)
    
  
 
    



if __name__ == "__main__":
    main()
    
