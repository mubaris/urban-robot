# Urban Robot

![Urban Robot](ub.png)

Reddit bot which replies to sarcastic comments


## Libraries

* numpy, scipy - For Mathematical and Scientific processes
* nltk - NLP Application
* scikit - Model Training and Feature Extraction
* textblob - Sentiment Analysis
* pickle - Pickling Models and Vectorizers
* langdetect - Language Detection of comments
* praw - Reddit Bot

## Features Used

* Sentiment Analysis of full text, equal 2 and 3 parts of text
* n-grams - 1 to 5
* Term Frequency–Inverse Document Frequency(TF-IFD) after stemming, tokenizing and using n-grams of 1 to 5
* Part of Speech Dictionary Vector
* Topic Modeling

## Data Preprocessing

* Removed URLs
* Removed Stopwords
* Removed words with less than 4 tokens

## Model Training and Classification

Using above Features and Preprocessing 4 models are trained,

* Logistic Regression
* Linear SVM
* SVM with Gaussian Kernel
* Random Forest

If a comment is predicted as 'sarcastic' by 3 out 4 models, it is treated as sarcastic.

## Files

* `classifier.py` - Training and Testing Models
* `bot.py` - Reddit Bot
* `main.ipynb` - iPython Notebook led to the final model hypothesis

## Running

1. Register for new Reddit App [here](https://www.reddit.com/prefs/apps/) and fill details (username, password, client id, client secret) under name 'bot1' in `praw.ini`

2. Run `classifier.py` with Python 3(Optional) or use pretrained models

3. Run `bot.py` with Python 3

That's it.

Logs can accessed at `comment.log`

How to fill [praw.ini](https://praw.readthedocs.io/en/v4.0.0/getting_started/configuration/prawini.html)

Final accuracy of models are in `final_accuracy.txt`

## Dataset

Dataset is available in `container`

Downloaded from [here](https://nlds.soe.ucsc.edu/sarcasm1)
