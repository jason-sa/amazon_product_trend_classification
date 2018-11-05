import AmazonReviews

PATH = '../data/amazon_reviews_us_Toys_v1_00.tsv'

def main():
    ar = AmazonReviews.AmazonReviews()
    print(ar.date_filter)
    ar.load_data(path=PATH)
    print(ar.reviews_df.head())

if __name__ == '__main__':
    main()
