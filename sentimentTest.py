import pickle
import glob
import csv
import pandas as pd


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

    # creating a new column for file names here.
    # it will take a list of files names
    df.insert(2, "file", file_names, True)
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df


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

    # Text to classify should be in a list.
    # text = ['I hate twitter',
    #         "I cannot wait for this Valorant tournament",
    #         "I can't say Minecraft is a good game",
    #         "I hope all League players disappear",
    #         "I think minecraft is a shitty game"]

    df = predict(vectoriser, LRmodel, lists_from_csv, file_names)
    print(df)