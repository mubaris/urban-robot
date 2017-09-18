from collections import Counter
import pickle
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix, hstack
import praw
import classifier

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def create_features(text):
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
    else:
        return False

reddit = praw.Reddit('bot1', user_agent='pyMubu.v0.1 (by /u/mubumbz)')

subreddit = reddit.subreddit('india')

comments = subreddit.stream.comments()

for comment in comments:
    text = comment.body
    if not classifier.is_too_short(text, 10):
        print(text)
        print()
        print(predictor(text))
        print()
        print()
        print("------------------------------")

lg_pkl.close()
svm_pkl.close()
linear_svm_pkl.close()
rf_pkl.close()