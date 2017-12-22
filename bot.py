import random
import sys
import time
import logging
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from langdetect import detect
import praw
import classifier

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    """stemmer helper"""
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    """returns stemmed tokens"""
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def create_features(text):
    """returns csr_matrix of all features for given text"""
    text = classifier.remove_url(text)
    text = classifier.remove_stopwords(text)

    sentiment_arr = csr_matrix(classifier.find_sentiment([text]))

    tfid_pkl = open('tfid.pkl', 'rb')
    tfidf = pickle.load(tfid_pkl)

    tfs_arr = tfidf.transform([text])

    pos_pkl = open('pos.pkl', 'rb')
    vec = pickle.load(pos_pkl)

    pos_arr = vec.transform(classifier.get_pos_features([text]))

    topic_pkl = open('topic.pkl', 'rb')
    lda = pickle.load(topic_pkl)

    topic_arr = lda.transform(tfs_arr)

    features = hstack([sentiment_arr, tfs_arr, pos_arr, topic_arr])

    tfid_pkl.close()
    pos_pkl.close()
    topic_pkl.close()

    return features

lg_pkl = open('logistic_regression.pkl', 'rb')
logistic_model = pickle.load(lg_pkl)
svm_pkl = open('svm.pkl', 'rb')
svm_model = pickle.load(svm_pkl)
linear_svm_pkl = open('linear_svm.pkl', 'rb')
linear_svm_model = pickle.load(linear_svm_pkl)
rf_pkl = open('rf.pkl', 'rb')
rf_model = pickle.load(rf_pkl)


def predictor(text):
    """returns boolean decision based on text is sarcasm or not"""
    features = create_features(text)
    out = []
    out.append(logistic_model.predict(features)[0])
    out.append(svm_model.predict(features)[0])
    out.append(linear_svm_model.predict(features)[0])
    out.append(rf_model.predict(features)[0])
    count = 0
    for i in range(4):
        if out[i] == 'sarc':
            count += 1
    if count > 2:
        return True
    return False

replies = ["PMSL", "ROFLMAO", "ROFLCOPTER", "LULZ", "BWAHAHA", "LOL", "LMAO", "ROFL", "OMG", ]

reddit = praw.Reddit('bot1', user_agent='pyMubu.v0.1 (by /u/mubumbz)')

subreddit = reddit.subreddit('india')

comments = subreddit.stream.comments()

stores_exception = None
count = 0

logging.basicConfig(filename='comments.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

for comment in comments:
    try:
        text = comment.body
        if not (classifier.is_too_short(text, 10)) and detect(text) == 'en':
            if predictor(text):
                message = random.choice(replies)
                comment.reply(message)
                count += 1
                time.sleep(120)
                info = text + '\n' + message + '\n' + '---------'
                logging.info(info)
    except KeyboardInterrupt:
        print("\nTotal Replies: ", count)
        sys.exit()

lg_pkl.close()
svm_pkl.close()
linear_svm_pkl.close()
rf_pkl.close()
