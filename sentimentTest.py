import pickle
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt

def load_models():
    # Load the vectoriser.
    file = open('models/vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()

    # Load the LR Model.
    file = open('models/LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel


def predict(vectoriser, model, text, file_names=None):
    # Predict the sentiment
    textdata = vectoriser.transform(text)
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])

    # Creating a new column for file names here.
    # it will take a list of files names
    df.insert(2, "file", file_names, True)
    df = df.replace([0, 1], ["Negative", "Positive"])

    report(df)

    return df

def report(df):


    # write tweet html to file
    tweetTable = df.to_html()
    text_file = open("Report.html", "w")
    text_file.write(' <center> ')
    text_file.write('<h1> Sentiment Analysis Report: Test Tweets </h1> <br>')
    text_file.write(' </center> ')

    text_file.write('<h2> Test Data </h2>')
    text_file.write(tweetTable+ '<br> ')

    # write sentiment frequency html to file
    text_file.write('<h2> Sentiment Count </h2>')
    count = df.groupby(['sentiment']).count()
    freq = count.to_html()
    text_file.write(freq+ ' <br> ')

    # write sentiment percentage html to file
    text_file.write('<h2> Sentiment Frequency </h2>')
    perc  = (df.groupby('sentiment').size() / df['sentiment'].count()) * 100
    perecent = perc.to_string()
    perecent = perecent.replace('sentiment', '')
    perecent = perecent.replace('Negative', '<b> Negative:  </b>')
    perecent = perecent.replace('Positive', '<b> Positive: </b>')
    text_file.write(perecent+ ' <br> <br> ')


    #print highest tweet with negative and positive sentiments
    text_file.write('<h2> 2 random tweets with a positive sentiment: </h2>')
    cl = df.loc[df.sentiment == "Positive", ['text']].sample(2).values
    for c in cl:
        text_file.write(c[0]+ ' <br> ')


    text_file.write('<h2> 2 random tweets with a negative sentiment: </h2>')
    cl = df.loc[df.sentiment == "Negative", ['text']].sample(2).values
    for c in cl:
        text_file.write(c[0]+ '<br> ')


    s = count.text
    fig, ax = plt.subplots()
    s.plot.pie()
    fig.savefig('graphs/my_plot.png')

    text_file.write(" <img src = "+ '"graphs/my_plot.png"'+' > <br> ')


    text_file.close()

if __name__ == "__main__":
    # Loading the models.
    vectoriser, LRmodel = load_models()

    path = './game_data/*.csv'
    lists_from_csv = []
    file_names = []
    for f in glob.glob(path):
        file = open(f, "r")
        csv_reader = csv.reader(file)
        for row in csv_reader:
            file_names.append(f)
            lists_from_csv.append(row[0])

    df = predict(vectoriser, LRmodel, lists_from_csv, file_names)

    print(df)