from flask import Flask, make_response, request, abort, jsonify, render_template
from api import get_best_comment

## Some weird error related to lambda function in the CountVectorizer
import re
from nltk import SnowballStemmer

stemmer = SnowballStemmer('english') # used in the model to predict whether or not a product will trend


app = Flask(__name__)
app.debug = True

@app.route('/trend_score', methods=['POST'])
def get_trend_score():
    if not request.json or ('review' not in request.json):
        abort(400)

    review = request.json['review']

    best_comment, best_score, orig_score, best_word = get_best_comment(review)

    best_score = str(round(best_score,2) * 100) + '%'
    orig_score = str(round(orig_score,2) * 100) + '%'

    word_pos = best_comment.find(best_word)
    word_len = len(best_word)
    best_comment = (best_comment[:word_pos] + 
                    '<span style="font-weight:bold; color:red">' + 
                    best_word + 
                    '</span>' + 
                    best_comment[(word_pos+word_len):])

    response = {
        'comment': review,
        'orig_score': orig_score,
        'best_comment': best_comment,
        'best_score': best_score
    }

    return jsonify(response), 201

@app.route('/')
def index():
    return render_template('home_with_styling.html')
   
if __name__ == '__main__':
    app.run(debug=False)
