from data_loader import load_and_clean_data
from visualisation import plot_interaction_overview
from AD_and_SM_analyse import run_agegroup_logit_regressions

def main():
    
    df = load_and_clean_data() 
    plot_interaction_overview(df) 
    run_agegroup_logit_regressions(df)
  
 
    



if __name__ == "__main__":
    main()
    
