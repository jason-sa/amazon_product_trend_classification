import AmazonReviews

PATH = '../data/amazon_reviews_us_Toys_v1_00.tsv'

def main():
    ar = AmazonReviews.AmazonReviews()
    # print(ar.date_filter)
    ar.load_data(path=PATH)
    ar.calc_trend_score()
    # # print(ar.reviews_df.head())
    print(ar.product_trend_df[ar.product_trend_df.trend == 1].describe())

if __name__ == '__main__':
    main()
