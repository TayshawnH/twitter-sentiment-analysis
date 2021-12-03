import re
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Importing the dataset from kaggle
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


def preprocess(textdata):
    processed_text = []

    # Create Lemmatizer.
    lemmatizer = WordNetLemmatizer()
    for tweet in textdata:
        tweet = tweet.lower()
        # Remove retweet text "RT"
        tweet = re.sub(r'^rt\s', '', tweet)
        # Remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#', '', tweet)
        # Remove punctuations
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # remove handles
        tweet = re.sub(r'@[^\s]+', '', tweet)
        tweet_words = ''
        for word in tweet.split():
            if len(word) > 1:
                # Lemmatizing the word.
                word = lemmatizer.lemmatize(word)
                tweet_words += (word + ' ')
        processed_text.append(tweet_words)

    return processed_text


t = time.time()
processedText = preprocess(text)
print(f'Text Preprocessing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')


X_train, X_test, y_train, y_test = train_test_split(processedText, sentiment,
                                                    test_size=0.2, random_state=0)
print(f'Data Split complete.')

vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
vectoriser.fit(X_train)
# vectors = vectoriser.fit_transform(X_train)
# words_df = pd.DataFrame(vectors.todense(), columns=vectoriser.get_feature_names())
print(f'Vectoriser has been fitted.')
# print(words_df.head())

X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)
# print the shape of the training and test data
print(X_train.shape)
print(X_test.shape)
print(f'Data Transformed.')


def model_evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # Compute the Confusion matrix and label the true and false positives.
    matrix = confusion_matrix(y_test, y_pred)
    labels = pd.Series(['Negative', 'Positive'])
    df_cm = pd.DataFrame(matrix, columns="Predicted " + labels, index="Is " + labels).div(matrix.sum(axis=1), axis=0)
    print(df_cm)


# Logistic Regression
# by default the L2 Regularization technique is used
# this is applied to avoid over-fitting
LRmodel = LogisticRegression(max_iter=1000, n_jobs=-1, C=2.0)

# Fit the model according to the given training data
# X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
# Training vector, where n_samples is the number of samples and n_features is the number of features.
# y_train : array-like of shape (n_samples)
# Target vector relative to X_train
# It is being trained on the training data. the Features and the labels
LRmodel.fit(X_train, y_train)
model_evaluate(LRmodel)

file = open('models/LR.pickle', 'wb')
pickle.dump(LRmodel, file)
file.close()

file = open('models/vectoriser-ngram-(1,2).pickle', 'wb')
pickle.dump(vectoriser, file)
file.close()