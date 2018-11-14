import spacy
import dill
import re
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import re

stemmer = SnowballStemmer('english') # used in the model to predict whether or not a product will trend
MODELING_PATH = '../data/modeling/'

def load(obj_name):
    f = MODELING_PATH + obj_name
    return dill.load(open(f, 'rb'))

# Load the word2vec model and final classifier model
nlp = spacy.load('en_core_web_lg')
final_model = load('final_full_model.pkl')
# final_model = load('final_full_model_test.pkl')

# build the vocabulary of trending reviews
review_corpus = load('review_corpus.pkl')
review_vocab = [nlp.vocab[w] for w in review_corpus]


def token_comment(comment):
    '''
    Tokenizes a comment and removes stop words. Builds the list of candidate words to be replaced.

    comment: string
    An Amazon toy review comment

    returns: list
    List of words toekenized from the comment
    '''
    tkpat = re.compile('\\b[a-z][a-z]+\\b')
    comment_token = tkpat.findall(comment)
    return [w for w in comment_token if w not in set(stopwords.words())]

def most_similar(word, top=10):
    '''
    Returns the top (default 10) similar words based on spaCy 'en_core_web_lg' model.

    word: string

    top: int
    The top n similar words to be returned

    returns: list
    List of top n words.
    '''
    by_similarity = sorted(review_vocab, key=lambda w: word.similarity(w), reverse=True)
    return [w.orth_ for w in by_similarity[:top]]

def get_best_comment(comment):
    ''' 
    Analyzes an Amazon Toy review comment and determines if any one word could be replaced to increase the
    probability of the product trending.

    comment: string
    An Amamzon toy review

    returns: (string, float, float) (best comment, best comment prob, orig comment prob)
    A tuple returning the comment with the highest probability, its related probability, and the probability of
    the original comment.
    '''
    comment_tokenized = token_comment(comment)
    comment_list = [comment]

    for t in comment_tokenized:
        sim_words = most_similar(nlp.vocab[t])
        for s in sim_words:
            new_comment = comment.replace(t, s)
            if new_comment != comment:
                comment_list.append(comment.replace(t, s))

    comment_probs = final_model.predict_proba(comment_list)[:,1]

    return comment_list[comment_probs.argmax()], comment_probs.max(), comment_probs[0]

if __name__ == '__main__':
    comment = 'This toy is amazing! So much worth the bucks!!'
    print(get_best_comment(comment))