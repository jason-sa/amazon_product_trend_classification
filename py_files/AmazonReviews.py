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

import seaborn as sns
import matplotlib.pyplot as plt

from SparseMatrixUtils import SparseMatrixUtils as smu

class AmazonReviews():
    ''' Class for reading Amazon review data and building a ML model to predict whether or not a product
        will trend based on a customer review. The review data is sourced from (https://s3.amazonaws.com/amazon-reviews-pds/readme.html).

        date_filter:
        reviews_df:
        reviews_selected_df:
        product_trend_df:
        obs:
        X:
        y:
        X_train: dictionary of training data sets with 'orig' being the original train/test split from X
    '''
    data_path = '../data/'
    RANDOM_STATE = 42
    _orig = 'orig'


    def __init__(self, date_filter=datetime(2014,1,1), pickle_path='../data/', name='AR'): # should add a flag to force to read from file
        ''' Initalizes an AmazonReview instance

        date_filter: (optional) 
        If None, then date_filter will be set to 2014-01-01
        '''
        self.date_filter = date_filter
        self.results = pd.DataFrame(columns=['Precision', 'Recall', 'F1', 'Accuracy','AUC'])
        self.y_scores = defaultdict(np.ndarray)
        self.pickle_path = pickle_path
        self.name = name

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

            Creates obs data frame (product_id, review_date, review_id, review_body, trend). 
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
    
    def create_train_test_split(self, train_reduction = 1):
        ''' Splits obs into X and y. Creates dictionaries to hold the train/test data sets, and performs an inital split.
        '''
        if train_reduction != 1:
            neg_split = int(self.obs.shape[0] * train_reduction * .99)
            pos_split = int(self.obs.shape[0] * train_reduction * .01)
            reduced_obs = pd.concat(
                [self.obs[self.obs['trend'] == 0].sample(n=neg_split),
                self.obs[self.obs['trend'] == 1].sample(n=pos_split)],
                axis=0
                )
        else:
            reduced_obs = self.obs

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

        # initalize the models dictionary to maintain the different trained models
        self.models = defaultdict(dict)
        self.pre_process_models = defaultdict(dict)

    def pre_process_data(self, p_model, p_name):
        ''' Pre-process the documents into DTM with sampling strategies applied. The pipeline and data sets are added to a dictionary.
        '''
        X_train_p, y_train_p = p_model.fit_sample(self.X_train, self.y_train)

        xt_name = 'X_train_'+ p_name
        smu.save_sparse_matrix(X_train_p, xt_name)

        self.pre_process_models[p_name].update(
            {
                'model': p_model,
                'X_train': xt_name, 
                'y_train': y_train_p
            }
        )

        # self.dump_models()

    def add_dtm(self, dtm_model, model_name):
        ''' Creates the document term matrix from the training data set

        dtm_model: sklearn model to create dtm
        model_name: name of the model 
        '''
        # X_train = dtm_model.fit_transform(self.models[self._orig]['X_train'])
        # X_test = dtm_model.transform(self.models[self._orig]['X_test'])

        # self.models[model_name].update({
        #     'model': dtm_model,
        #     'X_train': X_train,
        #     'X_test': X_test}
        # )

    def log_score(self, y_true, y_score, run_name, prob_cutoff = 0.5):
        '''
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
        '''
        '''
        plt.figure(figsize=(6,6))
        plt.plot([0,1],[0,1])

        for model, y_probs in self.y_scores.items():
            print(model)
            if model in 'TEST':
                print(model)
                fpr, tpr,_ = roc_curve(self.y_test, y_probs)    
            else:
                fpr, tpr,_ = roc_curve(self.y_train, y_probs)
        #     roc_auc = auc(fpr, tpr)
            plt.plot(fpr,tpr, label=model)
            
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    def run_model(self, model, model_name, p_name, sample=1):
        ''' Fits classification model, records accuracy, F1, precision, and recall to a data frame, and saves the confusion matrix
        '''
        X_train = smu.load_sparse_matrix(self.pre_process_models[p_name]['X_train'])
        y_train = self.pre_process_models[p_name]['y_train']

        if sample != 1:
            s = int(X_train.shape[0]*sample)
            i = np.random.choice(X_train.shape[0], size=s)
            X_train = X_train[i]
            y_train = y_train[i]

        model.fit(X_train, y_train) # fitted on upsampled data

        # transform original X_train to predict
        y_pred = model.predict(self.pre_process_models[p_name]['model'].steps[0][1].transform(self.X_train)) # need to predict on the original data

        scores = [
            accuracy_score(self.y_train, y_pred),
            precision_score(self.y_train, y_pred),
            recall_score(self.y_train, y_pred),
            f1_score(self.y_train, y_pred),
            roc_auc_score(self.y_train, y_pred)
        ]

        m_results = pd.DataFrame(data = scores, index= ['Accuracy', 'Precision','Recall','F1 Score', 'AUC'])
        m_results.columns = [model_name]

        if self.results is None:
            self.results = m_results
        else:
            self.results.drop(columns=[model_name], errors='ignore', inplace=True)
            self.results = pd.concat([self.results, m_results], axis=1)

        # store the best model, confusion matrix, and cv_results
        self.models[model_name].update(
            {
                'best_model': model.best_estimator_,
                'confusion_matrix': confusion_matrix(self.y_train, y_pred),
                'cv_results': model.cv_results_
            }
        )

        # save current state to pickle
        # self.dump_models(func=t_func)
    
    def conf_matrix(self, model_name):
        ''' Plots the confusion matrix for a specific model
        '''
        cm = self.models[model_name]['confusion_matrix']
        sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'], 
                    yticklabels=['actual_negative', 'actual_positive'], annot=True,
                    fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu")

    def dump_models(self, func=None):
        ''' Dumps the class into a pickle
        '''
        f = open(self.pickle_path+self.name+'.pkl', 'wb')
        pickle.dump(self, f)
        if func is not None:
            pickle.dump(func, f)

    def load_models(self):
        ''' Load a saved class from a pickle
        '''
        return pickle.load(open(self.pickle_path + self.name + '.pkl', 'rb'))

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

        
'''
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

first_pipe = imbPipeline([('cnt_v', CountVectorizer(stop_words='english', tokenizer=english_corpus, min_df=2)),
                          ('lda', LatentDirichletAllocation(n_jobs=-1, learning_method='online', random_state=42)),
                          ('sm', SMOTE(random_state=42)),
                          ('ss', StandardScaler()),
                          ('log_reg', LogisticRegression(random_state=42))])

params = {
    'lda__n_components': Integer(5, 20),
    'lda__learning_decay': Real(0.5, 1),
    'log_reg__C': Categorical([0.001,0.01,0.1,1,10,100])
}

grid = BayesSearchCV(first_pipe, params, n_jobs=-1)

grid.fit(ar.models['orig']['X_train'], ar.models['orig']['y_train'])

y_pred = grid.predict(ar.models['orig']['X_train'])

print('F1', f1_score(ar.models['orig']['y_train'], y_pred))
print('Precision',precision_score(ar.models['orig']['y_train'], y_pred))
print('Recall', recall_score(ar.models['orig']['y_train'], y_pred))
print(confusion_matrix(ar.models['orig']['y_train'], y_pred))

|metric|score|
---|---|
|F1| 0.03284926120870062|
|Precision| 0.016861979166666666|
|Recall| 0.6332518337408313|

||Pred No| Pred Yes|
|---|---|---|
|Act No| 25492| 15101|
|Act Yes|150|   259|

>>> import numpy as np
>>> from sklearn.preprocessing import FunctionTransformer
>>> transformer = FunctionTransformer(np.log1p, validate=True)
>>> X = np.array([[0, 1], [2, 3]])
>>> transformer.transform(X)
array([[0.        , 0.69314718],
       [1.09861229, 1.38629436]])
'''

