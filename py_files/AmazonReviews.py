import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle

class AmazonReviews:
    ''' Class for reading Amazon review data and building a ML model to predict whether or not a product
        will trend based on a customer review. The review data is sourced from (https://s3.amazonaws.com/amazon-reviews-pds/readme.html).

        date_filter:
        reviews_df:
        reviews_selected_df:
        product_trend_df:
    '''
    data_path = '../data/'

    def __init__(self, date_filter=datetime(2014,1,1)): # should add a flag to force to read from file
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
        # only load from file if pickle does not exist
        i = path.rfind('/')
        f = self.data_path + path[i+1:] + '.pkl'

        if os.path.isfile(f):
            self.reviews_df = pd.read_pickle(f)
            print('Read from pickle...')
        else: # 0.0955% of lines were not read due to errors
            self.reviews_df = pd.read_csv(path, sep='\t', error_bad_lines=False)
            
            with open(path, 'r') as f:
                lines = f.readlines()

            print()
            print(f'{1-self.reviews_df.shape[0] / len(lines):.4%} lines were not read due to data errors.')
            print()

            self.reviews_df['review_date'] = pd.to_datetime(self.reviews_df['review_date'], format='%Y-%m-%d')
            self.reviews_df.to_pickle('../data/amazon_reviews_us_Toys_v1_00.tsv.pkl')
            print('Saved to pickle...')

        # save data as pickle to reload if necessary
        self.reviews_df = self.reviews_df[self.reviews_df['review_date'] >= self.date_filter]
    
    def calc_trend_score(self, rating_power=1.5, review_days = 14, trend_percent = .99):
        ''' Calcualtes the trend scored defined as tanh( (review proportion * mean rating) / std rating ).
        The star rating can be smoothed by using rating_power. This will result in star_rating = star_rating ** rating_power.

        rating_power: default 1.5 
        Smooths the star rating scale 1-5 to star_rating ** rating_power
        
        review_time: default 14
        The trend score is calculated over the last n days since the last reivew. The default 30 days means
        all reviews that occured 30 days prior to the last review for the product are included in the trend score calcualtion. 

        trend_percent: default .99
        Expected values are (0,1). The precentile cut-off for what is trending or not. 99 meands the top 1% of the products will be identified as trending
        '''
        # smooth the star rating
        self.reviews_df['adj_star_rating'] = self.reviews_df['star_rating'] ** rating_power

        # create data frame for all reviews 30 days from the latest review
        last_review_df = self.reviews_df.groupby('product_id')['review_date'].max().reset_index()
        self.reviews_df = self.reviews_df.merge(last_review_df, how='inner', on='product_id')
        
        # drop max_review_date to avoid renaming
        self.reviews_df.drop(columns='max_review_date', errors='ignore', inplace=True)
        self.reviews_df.rename(columns={'review_date_x': 'review_date', 'review_date_y': 'max_review_date'}, inplace=True)
        self.reviews_selected_df = self.reviews_df[
            (self.reviews_df['max_review_date'] - self.reviews_df['review_date']) 
            <= timedelta(days=review_days)
            ]
        
        # calculate the review count, avg, and std star rating
        self.product_trend_df = self.reviews_selected_df.groupby('product_id')['adj_star_rating'].agg(['count','median','std'])
        self.product_trend_df['orig_std'] = self.product_trend_df['std']

        # set std with NA as the min std of the data set
        self.product_trend_df.loc[self.product_trend_df['std'] == 0, 'std'] = np.nan
        na_std = self.product_trend_df['std'].min()
        self.product_trend_df.fillna(na_std, inplace=True)

        # calcualte review success
        # total_reviews = self.product_trend_df['count'].sum()
        self.product_trend_df['review_success'] = (
            ( self.product_trend_df['count'] / review_days * self.product_trend_df['median'])
            / self.product_trend_df['std']
        )

        # calcualte the score and set the trend or not decision variable
        self.product_trend_df['trend_score'] = np.tanh(self.product_trend_df['review_success'])
        trend_cutoff = self.product_trend_df['trend_score'].quantile(trend_percent)
        self.product_trend_df['trend'] = (self.product_trend_df['trend_score'] > trend_cutoff).astype(int)

    def create_observations(self):
        ''' Creates the observation data set containing the first review and the unsupervised topic assigned.

        Notes: Need to get just one review or concatenate all reviews on the first day
        '''
        first_review_day = self.reviews_selected_df.groupby('product_id')['review_date'].min().reset_index()
        first_review_day = first_review_day.merge(
            self.reviews_selected_df.loc[:,['review_id', 'product_id', 'review_date']],
            how = 'inner',
            on = ['review_date', 'product_id']
        )

        first_review_day = first_review_day.groupby('product_id')['review_id'].min().reset_index()
        self.obs = first_review_day.merge(
            self.product_trend_df.reset_index(),
            how = 'inner',
            on = 
        )

        return NotImplementedError
    
    def create_train_test_split(self):
        ''' Performs train test split with optional sampling strategy
        '''
        raise NotImplementedError

    def create_dtm(self):
        ''' Creates the document term matrix from the training data set
        '''
        raise NotImplementedError
    
    def run_model(self):
        ''' Runs a single model
        '''
        return NotImplementedError
    
    def cross_validate(self):
        ''' Performs 10-fold CV 
        '''
        return NotImplementedError

    def model_metrics(self):
        ''' Builds heatmap and calcualtes accuracy, recall, precision, and F1
        '''
        return NotImplementedError

    def unsupervised_model(self):
        ''' Builds some unsupervised model
        '''
    
    def calc_inertia(self):
        ''' Runs some range of parameters and calcualtes inertia
        '''

        


