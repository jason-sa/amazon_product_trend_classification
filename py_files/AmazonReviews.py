import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import re
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from nltk.stem import SnowballStemmer

import matplotlib.pyplot as plt

class AmazonReviews():
    ''' Class for reading Amazon review data and building a ML model to predict whether or not a product
        will trend based on a customer review. The review data is sourced from (https://s3.amazonaws.com/amazon-reviews-pds/readme.html).

        date_filter: DataFrame 
        Filters the raw Amazon review data
        
        reviews_df: DataFrame
        Filtered data frame of the Amazon review data
        
        reviews_selected_df: DataFrame
        Filtered reviews_df for the time window to calculate the trend score
        
        product_trend_df: DataFrame
        Output of the trend calculation process and can analyze whether the trend score is calcualted correctly
        
        obs: DataFrame
        Entire set of observations the model will be trained and tested upon. 
        
        X: np.array
        Array for sklearn interface representing the feature space. 
        
        y: np.array
        Array for sklearn interface representing the target.
        
        X_train: np.array 
        Array for the sklearn interface representing the training feature space.       
 
        X_test: np.array
        Array for the sklearn interface representing the testing feature space.
        
        y_train: np.array
        Array for the sklearn interface representing the training target.
        
        y_test: np.array
        Array for the sklearn interface representing the testing target.
        
        results: DataFrame
        Stores the results of each model. DataFrame consists of accuracy, precision, recall, F1, and AUC.
        
        y_scores: defaultdict 
        Dictionary storing the target probabilities for each model.
    '''
    data_path = '../data/'
    RANDOM_STATE = 42


    def __init__(self, date_filter=datetime(2014,1,1)):# should add a flag to force to read from file
        ''' Initalizes an AmazonReview instance

        date_filter: (optional) 
        If None, then date_filter will be set to 2014-01-01
        '''
        self.date_filter = date_filter
        self.results = pd.DataFrame(columns=['Precision', 'Recall', 'F1', 'Accuracy','AUC'])
        self.y_scores = defaultdict(np.ndarray)

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
    
    def calc_trend_score(self, rating_power=1.5, review_days = 30, trend_percent = .99):
        ''' Calcualtes the trend scored defined as tanh( (review proportion * mean rating) / std rating ).
        The star rating can be smoothed by using rating_power. This will result in star_rating = star_rating ** rating_power.

        rating_power: default 1.5 
        Smooths the star rating scale 1-5 to star_rating ** rating_power
        
        review_time: default 30
        The trend score is calculated over the last n days since the last reivew. The default 30 days means
        all reviews that occured 30 days prior to the last review for the product are included in the trend score calcualtion. 

        trend_percent: default .99
        Expected values are (0,1). The precentile cut-off for what is trending or not. 99 meands the top 1% of the products will be identified as trending
        '''
        # smooth the star rating
        self.reviews_df['adj_star_rating'] = self.reviews_df['star_rating'] ** rating_power

        # create data frame for all reviews 'review_days' days from the first review
        first_review_df = self.reviews_df.groupby('product_id')['review_date'].min().reset_index()
        self.reviews_df = self.reviews_df.merge(first_review_df, how='inner', on='product_id')
        
        # drop max_review_date to avoid renaming
        self.reviews_df.drop(columns='min_review_date', errors='ignore', inplace=True)
        self.reviews_df.rename(columns={'review_date_x': 'review_date', 'review_date_y': 'min_review_date'}, inplace=True)
        self.reviews_selected_df = self.reviews_df[
            (self.reviews_df['review_date'] - self.reviews_df['min_review_date']) 
            >= timedelta(days=review_days)
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
        self.product_trend_df['trend'] = (self.product_trend_df['trend_score'] >= trend_cutoff).astype(int)

    def create_observations(self):
        ''' Creates the observation data set containing the first review and the unsupervised topic assigned.

            Creates obs data frame (product_id, review_date, review_id, review_body, trend, star rating). 
            The obs data frame combines the first review with the product trend. If a review body is empty, then the product is dropped.
        '''

        # get all reviews which appeared in the first day of the horizon
        first_review_day = self.reviews_selected_df.groupby('product_id')['review_date'].min().reset_index()
        first_review_day = first_review_day.merge(
            self.reviews_selected_df.loc[:,['review_id', 'product_id', 'review_date', 'review_body', 'star_rating']],
            how = 'inner',
            on = ['review_date', 'product_id']
        )
        # print(first_review_day.head())

        # get only one reivew if many occured on the first day
        first_review = first_review_day.groupby('product_id')['review_id'].head(1).reset_index()
        # print(first_review.head())
        self.obs = first_review_day.merge(
            first_review,
            how = 'inner',
            on = ['review_id']
        )

        self.obs = self.obs.merge(
            self.product_trend_df.reset_index().loc[:,['product_id', 'trend']],
            how = 'inner',
            on = 'product_id'
        )

        self.obs.drop(columns='index', inplace=True)
        self.obs.dropna(inplace=True)
    
    def create_train_test_split(self):
        ''' Cleans the review body text by removing digits and underscores. Splits obs into X and y. Creates dictionaries to hold the train/test data sets, and performs an inital split.
        '''

        self.X = (reduced_obs.review_body
                    .str.replace(r"""\w*\d\w*""", ' ')  # remove digits
                    .str.replace('_', ' ')              # remove underscores
        )

        self.y = reduced_obs.trend

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X, 
                                                            self.y, 
                                                            test_size = 0.25, 
                                                            stratify=self.y,
                                                            random_state = self.RANDOM_STATE)


    def log_score(self, y_true, y_score, run_name, prob_cutoff = 0.5):
        ''' Logs the related classification metrics for the training dataset.

        y_true: numpy.aray (nsamples,)
        Array of the actual classification.

        y_score: numpy.array(nsamples,)
        Probablity array from the trained model.

        run_name: string
        Name for the model being scored.

        prob_cutoff: float
        Probability cutoff for calcualting the confustion matrix related metrics 

        '''
        # log scores
        y_score_decision = (y_score >= 0.5).astype(int)

        run_results = {
            'Precision': precision_score(y_true, y_score_decision),
            'Recall': recall_score(y_true, y_score_decision),
            'F1': f1_score(y_true, y_score_decision),
            'Accuracy': accuracy_score(y_true, y_score_decision),
            'AUC': roc_auc_score(y_true, y_score)
        }

        self.results.drop(index=run_name, errors='ignore', inplace=True)
        self.results = self.results.append(pd.DataFrame(run_results, index=[run_name]))

        # save y_score for later calculations
        self.y_scores[run_name] = y_score

    def plot_roc_curve(self):
        ''' Creates a ROC curve plot for all models which have been logged.
        '''
        plt.figure(figsize=(6,6))
        plt.plot([0,1],[0,1])

        for model, y_probs in self.y_scores.items():
            fpr, tpr,_ = roc_curve(self.y_train, y_probs)
            plt.plot(fpr,tpr, label=model)
            
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

