from data_loader import load_and_clean_data 
from AD_interaction import(barplots_vs_ads, plot_ads_engagement_distribution, 
                           barplots_intent_vs_targets, barplots_discount_vs_targets, 
                           heatmap_return_by_category_and_ads, heatmap_ads_by_age_and_category,
                           heatmap_purchase_intent_by_age_and_category, plot_return_share_by_age_group,
                           plot_ads_users_count_by_age_group, plot_purchase_frequency_by_age, 
                           plot_purchase_amount_by_age, plot_total_revenue_by_age_group, 
                           heatmap_return_rate_by_location_and_category)
from visualisation import plot_interaction_overview


def main():
    
    df = load_and_clean_data()
    barplots_vs_ads(df)
    plot_interaction_overview(df)
    plot_ads_engagement_distribution(df)
    barplots_intent_vs_targets(df)
    barplots_discount_vs_targets(df)
    heatmap_return_by_category_and_ads(df)
    heatmap_ads_by_age_and_category(df)
    heatmap_purchase_intent_by_age_and_category(df)
    plot_return_share_by_age_group(df)
    plot_ads_users_count_by_age_group(df)
    plot_purchase_frequency_by_age(df)
    plot_purchase_amount_by_age(df)
    plot_total_revenue_by_age_group(df)
    heatmap_return_rate_by_location_and_category(df)



if __name__ == "__main__":
    main()
    
