import re
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
# un-comment both of these if this give an error after you run the program.
# import nltk
# nltk.download('wordnet')
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# Importing the dataset
DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('dataset/processed_tweet_dataset.csv',
                      encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Remove everything but sentiment and tweets
dataset = dataset[['sentiment', 'text']]
# Replacing the values to ease understanding. so that it can be O and 1's
dataset['sentiment'] = dataset['sentiment'].replace(4, 1)

# Storing data in lists.
text, sentiment = list(dataset['text']), list(dataset['sentiment'])

# this util is used in our twitter_data.py
def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def preprocess(textdata):
    processedText = []

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()

    for tweet in textdata:
        tweet = tweet.lower()
        clean_tweet(tweet)
        tweet_words = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            if len(word) > 1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweet_words += (word + ' ')

        processedText.append(tweet_words)

    return processedText

t = time.time()
processedtext = preprocess(text)
print(f'Text Preprocessing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')

X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment,
                                                    test_size = 0.05, random_state = 0)
print(f'Data Split done.')

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')

X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)
print(f'Data Transformed.')


def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # Compute the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    print(cf_matrix)


# Logistic Regression
# by default the L2 Regularization technique is used
# this is applied to avoid over-fitting
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)

file = open('models/LR.pickle', 'wb')
pickle.dump(LRmodel, file)
file.close()

file = open('models/vectoriser-ngram-(1,2).pickle', 'wb')
pickle.dump(vectoriser, file)
file.close()