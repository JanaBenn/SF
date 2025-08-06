from data_loader import load_and_clean_data
from visualisation import plot_interaction_overview
from SM_interaction import (analyze_and_plot_return_sm_correlation,
                            correlation_sm_return_overall, correlation_sm_return_furniture)
from SM_analyse import correlation_matrix_sm_return

def main():
    
    df = load_and_clean_data() 
    plot_interaction_overview(df) 
    analyze_and_plot_return_sm_correlation(df)
    correlation_sm_return_overall(df)
    correlation_sm_return_furniture(df)
    correlation_matrix_sm_return(df)
 
    



if __name__ == "__main__":
    main()
    
