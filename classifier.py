import re
from collections import Counter
import pickle
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

def remove_url(text):
    """returns text without the url"""
    pattern = "((http|ftp|https):\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w.,@?^=%&:\/~+#-])?"
    return re.sub(pattern, "", text)

def remove_stopwords(text):
    """returns string without english stopwords"""
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    filtered_sentence = " ".join([w for w in tokens if not w in stop_words])
    return filtered_sentence

def is_too_short(text, n):
    """returns true if the string is too short"""
    tokens = nltk.word_tokenize(text)
    return len(tokens) <= n

def divide_text(text, n):
    """returns list of strings with equal n parts"""
    tokens = nltk.word_tokenize(text)
    spilting_length = len(tokens) / n
    out = []
    x = 0
    for i in range(n):
        str_list = tokens[x:int(x+spilting_length)]
        string = " ".join(str_list)
        out.append(string)
        x = int(x+spilting_length)
    return out

def get_sentiment(arr):
    """returns sentiment polarity of list of strings"""
    n = len(arr)
    polar = []
    for i in range(n):
        analysis = TextBlob(arr[i])
        polar.append(analysis.sentiment.polarity)
    return polar

def find_sentiment(arr):
    """returns all sentiment features from a list of strings
        input string is divided in to equal parts of 1, 2 and 3.
        then the sentiment polarity of each parts are found
    """
    #n = len(arr)
    out = np.empty((len(arr), 6))
    for i in range(len(arr)):
        analysis = TextBlob(arr[i])
        uni_polar = analysis.sentiment.polarity
        string_list = divide_text(arr[i], 2)
        bi_polar = get_sentiment(string_list)
        string_list = divide_text(arr[i], 3)
        tri_polar = get_sentiment(string_list)
        out[i] = [uni_polar, bi_polar[0], bi_polar[1], tri_polar[0], tri_polar[1], tri_polar[2]]
    return out

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

def pos_tag_finder(text):
    """returns dictionary of parts of speech frequency"""
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    counts = Counter(tag for word, tag in tags)
    total = sum(counts.values())
    return dict((word, float(count)/total) for word, count in counts.items())

def get_pos_features(arr):
    """returns parts of speech features from a list"""
    out = np.array([])
    for i in range(len(arr)):
        pos_tags = pos_tag_finder(arr[i])
        out = np.append(out, pos_tags)
    return out

if __name__ == "__main__":

    # Getting Dataset
    dataset = load_files('container/', encoding="utf8", decode_error="replace")

    # Getting Text input and lables
    X = np.array([])
    y = np.array([])
    for i in range(len(dataset.data)):
        if not is_too_short(dataset.data[i], 4):
            noisless_text = remove_url(str(dataset.data[i]))
            noisless_text = remove_stopwords(noisless_text)
            X = np.append(X, noisless_text)
            if dataset.target[i] == 0:
                y = np.append(y, 'notsarc')
            else:
                y = np.append(y, 'sarc')
    
    # Splitting dataset in to training ans testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Finding setiment features of training and testing set
    sentiment_train = csr_matrix(find_sentiment(X_train))
    sentiment_test = csr_matrix(find_sentiment(X_test))

    # Initialize stemmer
    stemmer = PorterStemmer()

    # Finding TF-IDF features of training and testing set
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1, 5),
                            max_features=2440, min_df=0.05)
    tfs_train = tfidf.fit_transform(X_train)
    tfs_test = tfidf.transform(X_test)

    # Pickling TF_IDF Vectorizer
    tfid_pkl = open('tfid.pkl', 'wb')
    pickle.dump(tfidf, tfid_pkl)
    tfid_pkl.close()

    # Initialize Dictionary Vectorizer
    vec = DictVectorizer()

    # Finding Parts of Speech Features of training and testing set
    pos_train = vec.fit_transform(get_pos_features(X_train))
    pos_test = vec.transform(get_pos_features(X_test))

    # Pickling Parts of Speech DictVectorizer
    pos_pkl = open('pos.pkl', 'wb')
    pickle.dump(vec, pos_pkl)
    pos_pkl.close()

    # Initializing LDA
    lda = LatentDirichletAllocation(n_topics=10, learning_method='online')

    # Finding Topic Features of training and testing set
    topic_train = lda.fit_transform(tfs_train)
    topic_test = lda.transform(tfs_test)

    # Pickling LDA
    topic_pkl = open('topic.pkl', 'wb')
    pickle.dump(lda, topic_pkl)
    topic_pkl.close()

    # Stacking together all features for training and testing set
    final_train = hstack([sentiment_train, tfs_train, pos_train, topic_train])
    final_test = hstack([sentiment_test, tfs_test, pos_test, topic_test])

    lg_pkl_file = 'logistic_regression.pkl'
    lg_pkl = open(lg_pkl_file, 'wb')

    # Logistic Regression Model
    logistic_clf = LogisticRegression(C=0.2)
    logistic_clf = logistic_clf.fit(final_train, y_train)
    predict = logistic_clf.predict(final_test)
    pickle.dump(logistic_clf, lg_pkl)
    lg_pkl.close()
    print('Logistic = ', accuracy_score(y_test, predict))

    svm_pkl_file = 'svm.pkl'
    svm_pkl = open(svm_pkl_file, 'wb')

    # SVM Model with Gaussian Kernel
    svm_clf = SVC(C=4, gamma=0.1)
    svm_clf = svm_clf.fit(final_train, y_train)
    predict = svm_clf.predict(final_test)
    pickle.dump(svm_clf, svm_pkl)
    svm_pkl.close()
    print('SVM = ', accuracy_score(y_test, predict))

    linear_svm_pkl_file = 'linear_svm.pkl'
    linear_svm_pkl = open(linear_svm_pkl_file, 'wb')

    # Linear SVM Model
    linear_svm_clf = LinearSVC(C=0.1)
    linear_svm_clf = linear_svm_clf.fit(final_train, y_train)
    predict = linear_svm_clf.predict(final_test)
    pickle.dump(linear_svm_clf, linear_svm_pkl)
    linear_svm_pkl.close()
    print('lSVM = ', accuracy_score(y_test, predict))

    rf_pkl_file = 'rf.pkl'
    rf_pkl = open(rf_pkl_file, 'wb')

    # Random Forest Model
    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf = rf_clf.fit(final_train, y_train)
    predict = rf_clf.predict(final_test)
    pickle.dump(rf_clf, rf_pkl)
    rf_pkl.close()
    print('RF = ', accuracy_score(y_test, predict))
