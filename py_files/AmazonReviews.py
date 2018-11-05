import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AmazonReviews:
    ''' Class for reading Amazon review data and building a ML model to predict whether or not a product
        will trend based on a customer review. The review data is sourced from (https://s3.amazonaws.com/amazon-reviews-pds/readme.html).
    '''
    def __init__(self, date_filter=datetime(2014,1,1)):
        ''' Initalizes an AmazonReview instance

        date_filter: (optional) 
        If None, then date_filter will be set to 2014-01-01
        '''
        self.date_filter = date_filter
        self.reviews_df = pd.DataFrame() # df to hold the review data 

    def load_data(self, path):
        ''' Loads the AmazonReview data

        path: 
        File path to the tab separated Amazon Review data (https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
        '''
        self.reviews_df = pd.read_csv(path, sep='\t', error_bad_lines=False)
        
        with open(path, 'r') as f:
            lines = f.readlines()

        print()
        print(f'{1-self.reviews_df.shape[0] / len(lines):.4%} lines were not read due to data errors.')

        self.reviews_df['review_date'] = pd.to_datetime(self.reviews_df['review_date'], format='%Y-%m-%d')
        self.reviews_df = self.reviews_df[self.reviews_df['review_date'] >= self.date_filter]
    
    def calc_trend_score(self, rating_power=1.5, review_days = 30):
        ''' Calcualtes the trend scored defined as tanh( (review proportion * mean rating) / std rating ).
        The star rating can be smoothed by using rating_power. This will result in star_rating = star_rating ** rating_power.

        rating_power: default 1.5 
        Smooths the star rating scale 1-5 to star_rating ** rating_power
        
        review_time: default 30
        The trend score is calculated over the last n days since the last reivew. The default 30 days means
        all reviews that occured 30 days prior to the last review for the product are included in the trend score calcualtion. 
        '''

